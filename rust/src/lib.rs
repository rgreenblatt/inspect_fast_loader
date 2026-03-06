use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::io::Read;

// ---------------------------------------------------------------------------
// Helper: read a ZIP member into raw bytes
// ---------------------------------------------------------------------------

fn read_member_bytes(
    archive: &mut zip::ZipArchive<std::fs::File>,
    member_name: &str,
) -> Result<Vec<u8>, String> {
    let mut file = archive
        .by_name(member_name)
        .map_err(|e| format!("ZIP member not found: {member_name}: {e}"))?;
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read {member_name}: {e}"))?;
    Ok(buf)
}

fn open_archive(path: &str) -> PyResult<zip::ZipArchive<std::fs::File>> {
    let file = std::fs::File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
                "File not found: {path}"
            ))
        } else {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open {path}: {e}"))
        }
    })?;
    zip::ZipArchive::new(file)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid ZIP file: {e}")))
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Read an .eval format log file (ZIP archive) from disk.
///
/// Returns a Python dict with keys:
///   "header"          -> bytes (raw JSON of header.json or _journal/start.json)
///   "samples"         -> list[bytes] | None (raw JSON bytes per sample, or None if header_only)
///   "reductions"      -> bytes | None (raw JSON of reductions.json if present)
///   "has_header_json" -> bool (whether header.json was found)
///
/// JSON parsing is left to Python (json.loads) so NaN/Inf are handled natively.
#[pyfunction]
#[pyo3(signature = (path, header_only=false))]
fn read_eval_file(py: Python<'_>, path: &str, header_only: bool) -> PyResult<PyObject> {
    let mut archive = open_archive(path)?;

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
    let header_bytes = read_member_bytes(&mut archive, header_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e))?;
    result.set_item("header", PyBytes::new(py, &header_bytes))?;

    // Read reductions if present
    if entry_names.iter().any(|n| n == "reductions.json") {
        let reductions_bytes = read_member_bytes(&mut archive, "reductions.json")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        result.set_item("reductions", PyBytes::new(py, &reductions_bytes))?;
    } else {
        result.set_item("reductions", py.None())?;
    }

    // Read samples (unless header_only)
    if header_only {
        result.set_item("samples", py.None())?;
    } else {
        let sample_names: Vec<String> = entry_names
            .iter()
            .filter(|n| n.starts_with("samples/") && n.ends_with(".json"))
            .cloned()
            .collect();

        let samples_list = PyList::empty(py);
        for name in &sample_names {
            let bytes = read_member_bytes(&mut archive, name)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
            samples_list.append(PyBytes::new(py, &bytes))?;
        }

        result.set_item("samples", samples_list)?;
    }

    Ok(result.into_any().unbind())
}

/// Read headers from multiple .eval files in parallel using rayon.
///
/// Takes a list of file paths and returns a list of dicts, each with:
///   {header: bytes, samples: None, reductions: bytes|None, has_header_json: bool}
///
/// Uses rayon for true OS-level thread parallelism for ZIP decompression.
#[pyfunction]
fn read_eval_headers_batch(py: Python<'_>, paths: Vec<String>) -> PyResult<PyObject> {
    // Step 1: Read all headers in parallel (release GIL)
    let raw_results: Vec<Result<HeaderBytesResult, String>> = py.allow_threads(|| {
        use rayon::prelude::*;
        paths
            .par_iter()
            .map(|path| read_header_bytes_from_file(path))
            .collect()
    });

    // Step 2: Convert to Python objects (needs GIL)
    let result_list = PyList::empty(py);
    for (i, raw) in raw_results.into_iter().enumerate() {
        let hr = raw.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error reading {}: {e}",
                paths[i]
            ))
        })?;

        let dict = PyDict::new(py);
        dict.set_item("header", PyBytes::new(py, &hr.header_bytes))?;
        dict.set_item("samples", py.None())?;
        dict.set_item("has_header_json", hr.has_header_json)?;

        if let Some(red_bytes) = hr.reductions_bytes {
            dict.set_item("reductions", PyBytes::new(py, &red_bytes))?;
        } else {
            dict.set_item("reductions", py.None())?;
        }

        result_list.append(dict)?;
    }

    Ok(result_list.into_any().unbind())
}

struct HeaderBytesResult {
    header_bytes: Vec<u8>,
    has_header_json: bool,
    reductions_bytes: Option<Vec<u8>>,
}

fn read_header_bytes_from_file(path: &str) -> Result<HeaderBytesResult, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("Invalid ZIP {path}: {e}"))?;

    let entry_names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    let has_header_json = entry_names.iter().any(|n| n == "header.json");

    let header_name = if has_header_json {
        "header.json"
    } else {
        "_journal/start.json"
    };
    let header_bytes = read_member_bytes(&mut archive, header_name)?;

    let reductions_bytes = if entry_names.iter().any(|n| n == "reductions.json") {
        Some(read_member_bytes(&mut archive, "reductions.json")?)
    } else {
        None
    };

    Ok(HeaderBytesResult {
        header_bytes,
        has_header_json,
        reductions_bytes,
    })
}

/// Read a single sample from an .eval ZIP file by its entry name.
///
/// Returns the raw JSON bytes. The entry_name should be
/// e.g. "samples/1_epoch_1.json".
#[pyfunction]
fn read_eval_sample(py: Python<'_>, path: &str, entry_name: &str) -> PyResult<PyObject> {
    let mut archive = open_archive(path)?;
    let bytes = read_member_bytes(&mut archive, entry_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(e))?;
    Ok(PyBytes::new(py, &bytes).into_any().unbind())
}

/// Read summaries from an .eval ZIP file.
///
/// Returns the raw JSON bytes of the summaries (from summaries.json,
/// or concatenated from _journal/summaries/*.json entries).
#[pyfunction]
fn read_eval_summaries(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let mut archive = open_archive(path)?;

    let entry_names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();

    if entry_names.iter().any(|n| n == "summaries.json") {
        let bytes = read_member_bytes(&mut archive, "summaries.json")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        return Ok(PyBytes::new(py, &bytes).into_any().unbind());
    }

    // Fall back to _journal/summaries/*.json — return list of raw byte chunks
    let mut journal_summary_names: Vec<String> = entry_names
        .iter()
        .filter(|n| n.starts_with("_journal/summaries/") && n.ends_with(".json"))
        .cloned()
        .collect();
    journal_summary_names.sort_by_key(|n| {
        n.split('/')
            .last()
            .and_then(|f| f.strip_suffix(".json"))
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0)
    });

    let chunks = PyList::empty(py);
    for name in &journal_summary_names {
        let bytes = read_member_bytes(&mut archive, name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        chunks.append(PyBytes::new(py, &bytes))?;
    }

    Ok(chunks.into_any().unbind())
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
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(buf)
}

/// Python module definition.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_eval_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_eval_headers_batch, m)?)?;
    m.add_function(wrap_pyfunction!(read_eval_sample, m)?)?;
    m.add_function(wrap_pyfunction!(read_eval_summaries, m)?)?;
    m.add_function(wrap_pyfunction!(list_zip_entries, m)?)?;
    m.add_function(wrap_pyfunction!(read_zip_member, m)?)?;
    Ok(())
}
