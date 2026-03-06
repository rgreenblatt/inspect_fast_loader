use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse JSON bytes into a Python dict.
///
/// This is a minimal function to verify the Rust extension works.
/// Future implementations will add optimized log reading functions.
#[pyfunction]
fn parse_json_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let value: serde_json::Value =
        serde_json::from_slice(data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    json_value_to_py(py, &value)
}

/// Convert a serde_json::Value to a Python object.
fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for item in arr {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// List the entries of a ZIP file from bytes.
///
/// Returns a list of filenames in the ZIP archive.
#[pyfunction]
fn list_zip_entries(data: &[u8]) -> PyResult<Vec<String>> {
    use std::io::Cursor;
    let reader = Cursor::new(data);
    let archive = zip::ZipArchive::new(reader)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(archive.file_names().map(|s| s.to_string()).collect())
}

/// Read a specific member from a ZIP archive given as bytes.
///
/// Returns the uncompressed content of the specified member as bytes.
#[pyfunction]
fn read_zip_member(data: &[u8], member_name: &str) -> PyResult<Vec<u8>> {
    use std::io::{Cursor, Read};
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
    m.add_function(wrap_pyfunction!(list_zip_entries, m)?)?;
    m.add_function(wrap_pyfunction!(read_zip_member, m)?)?;
    Ok(())
}
