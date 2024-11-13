use crate::tensor::tensor::TensorView;
use crate::tensor::{dtype::Dtype, order::Order};
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read};

#[derive(Debug, Serialize, Deserialize)]
struct TensorMeta {
    name: String,
    offset: u64,
    size: u64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub(crate) timestamp: u64,
}

/// Represents a collection of named tensors
/// The on-disk format is:
/// - Header:
///     - Metadata:
///         - Timestamp (8 bytes)
///     - Number of tensors (4 bytes)
/// - Metadata Index:
///     - For each tensor:
///         - Name length (4 bytes)
///         - Name (variable length)
///         - Offset (8 bytes)
///         - Size (8 bytes)
/// - Data:
///     - Concatenated tensor data
pub struct Snapshot<'data> {
    meta: SnapshotMeta,
    tensors: HashMap<String, TensorView<'data>>,
}

impl<'data> Snapshot<'data> {
    pub fn new() -> Self {
        Self {
            meta: SnapshotMeta {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            tensors: HashMap::new(),
        }
    }

    pub fn add_tensor(&mut self, name: String, tensor: TensorView<'data>) {
        self.tensors.insert(name, tensor);
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorView<'data>> {
        self.tensors.get(name)
    }

    pub fn from_bytes(bytes: &'data [u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        // Read metadata
        let timestamp = cursor.read_u64::<LittleEndian>()?;
        let meta = SnapshotMeta { timestamp };

        let num_tensors = cursor.read_u32::<LittleEndian>()?;

        let mut tensors = HashMap::new();
        let mut tensor_metas = Vec::new();

        // Read metadata index
        for _ in 0..num_tensors {
            let name_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)?;

            let offset = cursor.read_u64::<LittleEndian>()?;
            let size = cursor.read_u64::<LittleEndian>()?;

            tensor_metas.push(TensorMeta { name, offset, size });
        }

        // Read tensor data using the metadata
        let data_start = cursor.position() as usize;
        for meta in tensor_metas {
            let tensor_data = &bytes[data_start + meta.offset as usize
                ..data_start + (meta.offset + meta.size) as usize];
            let tensor = TensorView::from_bytes(tensor_data)?;
            tensors.insert(meta.name, tensor);
        }

        Ok(Self { meta, tensors })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write metadata
        bytes.extend_from_slice(&self.meta.timestamp.to_le_bytes());

        // Write number of tensors
        bytes.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());

        // Write metadata index
        let mut offset = 0;
        let mut tensor_metas = Vec::new();
        for (name, tensor) in &self.tensors {
            let size = tensor.to_bytes().len() as u64;
            tensor_metas.push(TensorMeta {
                name: name.clone(),
                offset,
                size,
            });
            offset += size;
        }

        for meta in &tensor_metas {
            bytes.extend_from_slice(&(meta.name.len() as u32).to_le_bytes());
            bytes.extend_from_slice(meta.name.as_bytes());
            bytes.extend_from_slice(&meta.offset.to_le_bytes());
            bytes.extend_from_slice(&meta.size.to_le_bytes());
        }

        // Write tensor data
        for meta in &tensor_metas {
            let tensor = self.tensors.get(&meta.name).unwrap();
            bytes.extend_from_slice(tensor.to_bytes().as_slice());
        }

        bytes
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_basic() {
        let mut snapshot = Snapshot::new();

        // Create a simple tensor
        let tensor = TensorView::new(
            Dtype::F32,
            vec![2, 2],
            Order::RowMajor,
            vec![2, 2],
            vec![vec![0, 0]],
            &[0u8; 16],
        );

        snapshot.add_tensor("test_tensor".to_string(), tensor);

        assert!(snapshot.get_tensor("test_tensor").is_some());
        assert!(snapshot.get_tensor("nonexistent").is_none());

        // Test serialization with metadata
        let bytes = snapshot.to_bytes();
        let decoded = Snapshot::from_bytes(&bytes).unwrap();
        assert!(decoded.get_tensor("test_tensor").is_some());
        assert_eq!(decoded.meta.timestamp, snapshot.meta.timestamp);
    }
}
