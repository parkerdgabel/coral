use anyhow::Result;

use crate::tensor::dtype::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkMetadata {
    pub(crate) dtype: Dtype,
    pub(crate) number_of_elements: usize,
    pub(crate) number_of_bytes: usize,
}

impl ChunkMetadata {
    pub fn new(dtype: Dtype, number_of_elements: usize) -> Self {
        let number_of_bytes = number_of_elements * dtype.size();
        Self {
            dtype,
            number_of_elements,
            number_of_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk<'data> {
    metadata: ChunkMetadata,
    data: &'data [u8],
}

/// The chunk file format is:
/// - Header:
///   - Magic number (8 bytes): "CORALCHK"
///   - Version (4 bytes): u32 little endian
///   - Dtype (4 bytes): u32 little endian
///   - Number of elements (8 bytes): u64 little endian
/// - Data:
///   - Raw bytes of the elements
///
/// Total header size: 24 bytes
impl<'data> Chunk<'data> {
    pub fn new(metadata: ChunkMetadata, data: &'data [u8]) -> Self {
        Self { metadata, data }
    }

    pub fn metadata(&self) -> &ChunkMetadata {
        &self.metadata
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24 + self.data.len());
        bytes.extend_from_slice(b"CORALCHK");
        bytes.extend_from_slice(&(1u32).to_le_bytes());
        bytes.extend_from_slice(&self.metadata.dtype.to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.number_of_elements as u64).to_le_bytes());
        bytes.extend_from_slice(self.data);
        bytes
    }

    pub fn from_bytes(bytes: &'data [u8]) -> Result<Self> {
        if bytes.len() < 24 {
            anyhow::bail!("Invalid chunk file: too small");
        }
        if &bytes[0..8] != b"CORALCHK" {
            anyhow::bail!("Invalid chunk file: wrong magic number");
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into()?);
        if version != 1 {
            anyhow::bail!("Unsupported chunk version: {}", version);
        }
        let dtype = Dtype::from_le_bytes(bytes[12..16].try_into()?);
        let number_of_elements = u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
        let metadata = ChunkMetadata::new(dtype, number_of_elements);
        let data = &bytes[24..];
        Ok(Self::new(metadata, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_chunk_metadata() {
        let metadata = ChunkMetadata::new(Dtype::F32, 100);
        assert_eq!(metadata.dtype, Dtype::F32);
        assert_eq!(metadata.number_of_elements, 100);
        assert_eq!(metadata.number_of_bytes, 400);
    }

    #[test]
    fn test_chunk_serialization() {
        let data = vec![1u8, 2, 3, 4];
        let metadata = ChunkMetadata::new(Dtype::U16, 8);
        let chunk = Chunk::new(metadata, &data);

        let bytes = chunk.to_bytes();
        let decoded = Chunk::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.metadata(), chunk.metadata());
        assert_eq!(decoded.data(), chunk.data());
    }

    #[test]
    fn test_invalid_chunk_data() {
        // Too small
        assert!(Chunk::from_bytes(&[0u8; 23]).is_err());

        // Wrong magic number
        let invalid = vec![0u8; 100];
        assert!(Chunk::from_bytes(&invalid).is_err());

        // Wrong version
        let mut bytes = vec![0u8; 100];
        bytes[0..8].copy_from_slice(b"CORALCHK");
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(Chunk::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_empty_chunk() {
        let metadata = ChunkMetadata::new(Dtype::F64, 0);
        let chunk = Chunk::new(metadata, &[]);
        let bytes = chunk.to_bytes();
        let decoded = Chunk::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.metadata(), chunk.metadata());
        assert_eq!(decoded.data().len(), 0);
    }
}
