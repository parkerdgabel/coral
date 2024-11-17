use lib_coral::snapshot::Snapshot;
use lib_coral::tensor::{dtype::Dtype, order::Order, tensor::View};
use numpy::{ndarray, IxDyn, PyArray};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyDict};
use pyo3::{sync::GILOnceCell, types::PyModule, Py};
use pyo3::{FromPyObject, IntoPy, PyObject, Python};
use std::borrow::Cow;
use std::collections::HashMap;

static TORCH_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static NUMPY_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static TENSORFLOW_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();

use pyo3::prelude::*;

pub fn convert_numpy_array_to_tensor(
    py: Python,
    array: &PyArray<f32, IxDyn>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("dtype", array.dtype().name())?;
    dict.set_item("shape", array.shape())?;
    dict.set_item("data", PyBytes::new(py, array))?;
    dict.set_item("cell_order", "ROW_MAJOR")?;
    dict.set_item("tile_extent", array.strides())?;
    Ok(dict.into_py(py))
}

struct PyView {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
    cell_order: Order,
    tile_extent: Vec<usize>,
}

impl View for PyView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }

    fn cell_order(&self) -> Order {
        self.cell_order
    }

    fn tile_extent(&self) -> &[usize] {
        &self.tile_extent
    }
}

fn tensor_to_py_view(py: Python, tensor_dict: PyObject) -> PyResult<PyView> {
    let dict: &PyDict = tensor_dict.downcast(py)?;

    let shape: Vec<usize> = dict
        .get_item("shape")?
        .ok_or_else(|| PyValueError::new_err("Missing 'shape'"))?
        .extract()?;

    let dtype_str: String = dict
        .get_item("dtype")?
        .ok_or_else(|| PyValueError::new_err("Missing 'dtype'"))?
        .extract()?;

    let dtype = match dtype_str.as_str() {
        "BOOL" => Dtype::BOOL,
        "I8" => Dtype::I8,
        "U8" => Dtype::U8,
        "I16" => Dtype::I16,
        "U16" => Dtype::U16,
        "I32" => Dtype::I32,
        "U32" => Dtype::U32,
        "I64" => Dtype::I64,
        "U64" => Dtype::U64,
        "F16" => Dtype::F16,
        "F32" => Dtype::F32,
        "F64" => Dtype::F64,
        "BF16" => Dtype::BF16,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported dtype: {}",
                dtype_str
            )))
        }
    };

    let data: &PyBytes = dict
        .get_item("data")?
        .ok_or_else(|| PyValueError::new_err("Missing 'data'"))?
        .downcast()?;

    let cell_order: String = dict
        .get_item("cell_order")?
        .ok_or_else(|| PyValueError::new_err("Missing 'cell_order'"))?
        .extract()?;
    let cell_order = match cell_order.as_str() {
        "ROW_MAJOR" => Order::RowMajor,
        "COLUMN_MAJOR" => Order::ColumnMajor,
        _ => return Err(PyValueError::new_err("Invalid cell_order")),
    };

    let tile_extent: Vec<usize> = dict
        .get_item("tile_extent")?
        .ok_or_else(|| PyValueError::new_err("Missing 'tile_extent'"))?
        .extract()?;

    Ok(PyView {
        shape,
        dtype,
        data: data.as_bytes().to_vec(),
        cell_order,
        tile_extent,
    })
}

/// Create a snapshot from a dictionary of tensors.
///
/// Args:
///     tensor_dict (`Dict[str, Any]`):
///         Dictionary mapping tensor names to tensor objects (numpy arrays, torch tensors, etc.)
///         The dict is like {"tensor_name": {"dtype": "F32", "shape": [2, 3], "tile_extent": [1,1], "tile_order": "ROW_MAJOR", "cell_order": ROW_MAJOR, "data": b"\0\0"}}
///     chunk_size (`int`, *optional*):
///         Size of chunks to use for compression
///
/// Returns:
///     (`str`):
///         The snapshot hash
#[pyfunction]
#[pyo3(signature = (tensor_dict, chunk_size=None))]
fn create_snapshot(
    tensor_dict: HashMap<String, PyObject>,
    chunk_size: Option<usize>,
) -> PyResult<String> {
    Python::with_gil(|py| {
        let mut data = Vec::new();
        for (name, tensor) in tensor_dict {
            let py_view = tensor_to_py_view(py, tensor)?;
            data.push((name, py_view));
        }

        let config = if let Some(size) = chunk_size {
            lib_coral::SnapshotConfig::with_chunk_strategy(Box::new(
                lib_coral::chunk::strategy::FixedChunk::new(size),
            ))
        } else {
            lib_coral::SnapshotConfig::new()
        };

        lib_coral::snapshot(data, config).map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

#[pyclass]
struct PySnapshot {
    inner: Snapshot,
}

#[pymethods]
impl PySnapshot {
    #[new]
    fn new() -> Self {
        Self {
            inner: Snapshot::new(),
        }
    }

    fn get_tensor(&self, name: &str) -> Option<PyObject> {
        Python::with_gil(|py| {
            self.inner.get_tensor(name).map(|tensor| {
                let dict = PyDict::new(py);
                dict.set_item("dtype", tensor.dtype().to_string()).unwrap();
                dict.set_item("shape", tensor.shape()).unwrap();
                dict.set_item("data", PyBytes::new(py, &tensor.data()))
                    .unwrap();
                dict.set_item("cell_order", tensor.cell_order().to_string())
                    .unwrap();
                dict.set_item("tile_extent", tensor.tile_extent()).unwrap();
                dict.into_py(py)
            })
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names().cloned().collect()
    }
}

impl From<Snapshot> for PySnapshot {
    fn from(snapshot: Snapshot) -> Self {
        Self { inner: snapshot }
    }
}

/// Load a snapshot from its hash.
///
/// Args:
///     snapshot_hash (`str`):
///         The hash of the snapshot to load
///
/// Returns:
///     (`Snapshot`):
///         The loaded snapshot object
#[pyfunction]
fn load_snapshot(snapshot_hash: &str) -> PyResult<PySnapshot> {
    lib_coral::load_snapshot(snapshot_hash)
        .map(PySnapshot::from)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Create a commit from a snapshot.
///
/// Args:
///     snapshot_hash (`str`):
///         Hash of the snapshot to commit
///     parents (`List[str]`):
///         List of parent commit hashes
///     author (`str`):
///         Name of the commit author
///     email (`str`):
///         Email of the commit author
///     message (`str`):
///         Commit message
///
/// Returns:
///     (`str`):
///         The commit hash
#[pyfunction]
fn create_commit(
    snapshot_hash: &str,
    parents: Vec<String>,
    author: &str,
    email: &str,
    message: &str,
) -> PyResult<String> {
    lib_coral::commit(snapshot_hash, parents, author, email, message)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn coral_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(load_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(create_commit, m)?)?;
    m.add_class::<PySnapshot>()?;

    Ok(())
}
