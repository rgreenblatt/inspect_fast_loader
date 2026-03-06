use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};
use std::io::Read;

// ---------------------------------------------------------------------------
// NaN/Inf pre-processing
// ---------------------------------------------------------------------------

/// Replace NaN, Infinity, -Infinity tokens in JSON bytes with sentinel strings,
/// returning the modified bytes and a flag indicating whether any replacements were made.
///
/// The sentinels are chosen to be valid JSON strings that are extremely unlikely
/// to appear in real data:
///   NaN       -> "__NaN_SENTINEL__"
///   Infinity  -> "__Inf_SENTINEL__"
///   -Infinity -> "__NegInf_SENTINEL__"
///
/// We scan byte-by-byte, skipping over JSON strings (respecting escape sequences).
fn preprocess_nan_inf(input: &[u8]) -> (Vec<u8>, bool) {
    let mut output = Vec::with_capacity(input.len());
    let mut i = 0;
    let mut had_replacements = false;

    while i < input.len() {
        let b = input[i];

        // Skip over JSON strings
        if b == b'"' {
            output.push(b);
            i += 1;
            while i < input.len() {
                let c = input[i];
                output.push(c);
                i += 1;
                if c == b'\\' && i < input.len() {
                    // Push escaped character
                    output.push(input[i]);
                    i += 1;
                } else if c == b'"' {
                    break;
                }
            }
            continue;
        }

        // Check for NaN (must not be preceded by alphanumeric/underscore)
        if b == b'N' && i + 2 < input.len()
            && input[i + 1] == b'a' && input[i + 2] == b'N'
            && (i == 0 || !is_ident_char(input[i - 1]))
            && (i + 3 >= input.len() || !is_ident_char(input[i + 3]))
        {
            output.extend_from_slice(b"\"__NaN_SENTINEL__\"");
            i += 3;
            had_replacements = true;
            continue;
        }

        // Check for -Infinity
        if b == b'-' && i + 9 <= input.len()
            && &input[i + 1..i + 9] == b"Infinity"
            && (i == 0 || !is_ident_char(input[i - 1]))
            && (i + 9 >= input.len() || !is_ident_char(input[i + 9]))
        {
            output.extend_from_slice(b"\"__NegInf_SENTINEL__\"");
            i += 9;
            had_replacements = true;
            continue;
        }

        // Check for Infinity (positive)
        if b == b'I' && i + 8 <= input.len()
            && &input[i..i + 8] == b"Infinity"
            && (i == 0 || !is_ident_char(input[i - 1]))
            && (i + 8 >= input.len() || !is_ident_char(input[i + 8]))
        {
            output.extend_from_slice(b"\"__Inf_SENTINEL__\"");
            i += 8;
            had_replacements = true;
            continue;
        }

        output.push(b);
        i += 1;
    }

    (output, had_replacements)
}

#[inline]
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// JSON -> Python conversion
// ---------------------------------------------------------------------------

/// Convert a serde_json::Value to a Python object, restoring NaN/Inf sentinels.
fn json_value_to_py(py: Python<'_>, value: &serde_json::Value, restore_nan_inf: bool) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                // Fallback for very large integers
                let s = n.to_string();
                let int_val = s.parse::<i128>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Cannot parse number: {e}"))
                })?;
                Ok(int_val.into_pyobject(py)?.into_any().unbind())
            }
        }
        serde_json::Value::String(s) => {
            if restore_nan_inf {
                match s.as_str() {
                    "__NaN_SENTINEL__" => {
                        return Ok(PyFloat::new(py, f64::NAN).into_any().unbind());
                    }
                    "__Inf_SENTINEL__" => {
                        return Ok(PyFloat::new(py, f64::INFINITY).into_any().unbind());
                    }
                    "__NegInf_SENTINEL__" => {
                        return Ok(PyFloat::new(py, f64::NEG_INFINITY).into_any().unbind());
                    }
                    _ => {}
                }
            }
            Ok(s.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item, restore_nan_inf)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_value_to_py(py, v, restore_nan_inf)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Parse JSON bytes (with NaN/Inf support) into a Python dict/list/value.
///
/// Fast path: tries standard serde_json parsing first. Only falls back to
/// NaN/Inf preprocessing if standard parsing fails (most files don't contain
/// NaN/Inf, so this avoids the preprocessing overhead in the common case).
fn parse_json_with_nan_inf(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    // Fast path: try standard parsing first
    match serde_json::from_slice::<serde_json::Value>(data) {
        Ok(value) => return json_value_to_py(py, &value, false),
        Err(standard_err) => {
            // Standard parsing failed — try with NaN/Inf preprocessing
            let (processed, had_replacements) = preprocess_nan_inf(data);
            if had_replacements {
                let value: serde_json::Value = serde_json::from_slice(&processed)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("JSON parse error: {e}")))?;
                json_value_to_py(py, &value, true)
            } else {
                // No NaN/Inf found — original error is the real error
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("JSON parse error: {standard_err}")))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Parse JSON bytes into a Python object (dict, list, etc).
/// Supports NaN, Infinity, -Infinity as bare tokens.
#[pyfunction]
fn parse_json_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    parse_json_with_nan_inf(py, data)
}

/// Read a .json format log file from disk and parse it into a Python dict.
/// Supports NaN/Infinity/-Infinity in the JSON.
#[pyfunction]
fn read_json_file(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("File not found: {path}"))
        } else {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read {path}: {e}"))
        }
    })?;
    parse_json_with_nan_inf(py, &data)
}

/// Read an .eval format log file (ZIP archive) from disk.
///
/// Returns a Python dict with keys:
///   "header" -> dict (parsed header.json or _journal/start.json)
///   "samples" -> list[dict] | None (parsed samples/*.json, or None if header_only)
///   "reductions" -> list[dict] | None (parsed reductions.json if present)
///   "has_header_json" -> bool (whether header.json was found)
///
/// If header_only=True, only the header (and reductions) are read.
#[pyfunction]
#[pyo3(signature = (path, header_only=false))]
fn read_eval_file(py: Python<'_>, path: &str, header_only: bool) -> PyResult<PyObject> {
    let file = std::fs::File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("File not found: {path}"))
        } else {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open {path}: {e}"))
        }
    })?;

    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid ZIP file: {e}")))?;

    let result = PyDict::new(py);

    // Collect entry names
    let entry_names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    let has_header_json = entry_names.iter().any(|n| n == "header.json");

    result.set_item("has_header_json", has_header_json)?;

    // Read header
    let header_name = if has_header_json {
        "header.json"
    } else {
        "_journal/start.json"
    };
    let header_data = read_and_parse_member(py, &mut archive, header_name)?;
    result.set_item("header", header_data)?;

    // Read reductions if present
    if entry_names.iter().any(|n| n == "reductions.json") {
        let reductions_data = read_and_parse_member(py, &mut archive, "reductions.json")?;
        result.set_item("reductions", reductions_data)?;
    } else {
        result.set_item("reductions", py.None())?;
    }

    // Note: summaries.json is not parsed here as the Python code doesn't use it.
    // EvalLog doesn't have a top-level summaries field; summaries are only used
    // for the read_eval_log_sample_summaries function which we don't patch.

    // Read samples (unless header_only)
    if header_only {
        result.set_item("samples", py.None())?;
    } else {
        let samples_list = PyList::empty(py);

        // Collect sample entry names first (to avoid borrow issues)
        let sample_names: Vec<String> = entry_names
            .iter()
            .filter(|n| n.starts_with("samples/") && n.ends_with(".json"))
            .cloned()
            .collect();

        for name in &sample_names {
            let sample_data = read_and_parse_member(py, &mut archive, name)?;
            samples_list.append(sample_data)?;
        }

        result.set_item("samples", samples_list)?;
    }

    Ok(result.into_any().unbind())
}

/// Read and parse a single ZIP member as JSON with NaN/Inf support.
fn read_and_parse_member(
    py: Python<'_>,
    archive: &mut zip::ZipArchive<std::fs::File>,
    member_name: &str,
) -> PyResult<PyObject> {
    let mut file = archive.by_name(member_name).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("ZIP member not found: {member_name}: {e}"))
    })?;
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read {member_name}: {e}")))?;
    parse_json_with_nan_inf(py, &buf)
}

/// List the entries of a ZIP file from bytes.
#[pyfunction]
fn list_zip_entries(data: &[u8]) -> PyResult<Vec<String>> {
    use std::io::Cursor;
    let reader = Cursor::new(data);
    let archive = zip::ZipArchive::new(reader)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(archive.file_names().map(|s| s.to_string()).collect())
}

/// Read a specific member from a ZIP archive given as bytes.
#[pyfunction]
fn read_zip_member(data: &[u8], member_name: &str) -> PyResult<Vec<u8>> {
    use std::io::Cursor;
    let reader = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let mut file = archive
        .by_name(member_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e.to_string()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(buf)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_json_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_json_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_eval_file, m)?)?;
    m.add_function(wrap_pyfunction!(list_zip_entries, m)?)?;
    m.add_function(wrap_pyfunction!(read_zip_member, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_nan_basic() {
        let input = br#"{"x": NaN, "y": Infinity, "z": -Infinity}"#;
        let (output, had) = preprocess_nan_inf(input);
        assert!(had);
        let s = String::from_utf8(output).unwrap();
        assert!(s.contains("\"__NaN_SENTINEL__\""));
        assert!(s.contains("\"__Inf_SENTINEL__\""));
        assert!(s.contains("\"__NegInf_SENTINEL__\""));
        // Should be valid JSON now
        let _: serde_json::Value = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn test_preprocess_nan_in_string_untouched() {
        let input = br#"{"msg": "the value is NaN or Infinity"}"#;
        let (output, had) = preprocess_nan_inf(input);
        assert!(!had);
        assert_eq!(input.as_slice(), output.as_slice());
    }

    #[test]
    fn test_preprocess_no_nan() {
        let input = br#"{"x": 1, "y": "hello"}"#;
        let (output, had) = preprocess_nan_inf(input);
        assert!(!had);
        assert_eq!(input.as_slice(), output.as_slice());
    }

    #[test]
    fn test_preprocess_nan_in_array() {
        let input = br#"[NaN, 1, Infinity, -Infinity]"#;
        let (output, had) = preprocess_nan_inf(input);
        assert!(had);
        let s = String::from_utf8(output).unwrap();
        let _: serde_json::Value = serde_json::from_str(&s).unwrap();
    }
}
