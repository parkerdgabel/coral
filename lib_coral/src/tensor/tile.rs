use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    chunk::chunk::Chunk,
    store::{ObjectType, Store},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tile {
    // SHA-1 hashes are 40 characters in hex format
    chunk_refs: Vec<String>,
}

impl Tile {
    pub fn new(chunk_refs: Vec<String>) -> Self {
        // Validate that all refs are SHA-1 hashes
        debug_assert!(
            chunk_refs
                .iter()
                .all(|hash| hash.len() == 40 && hash.chars().all(|c| c.is_ascii_hexdigit())),
            "All chunk references must be valid SHA-1 hashes"
        );

        Self { chunk_refs }
    }

    pub fn chunk_refs(&self) -> &[String] {
        &self.chunk_refs
    }

    pub fn add_chunk_ref(&mut self, hash: String) {
        debug_assert!(
            hash.len() == 40 && hash.chars().all(|c| c.is_ascii_hexdigit()),
            "Chunk reference must be a valid SHA-1 hash"
        );
        self.chunk_refs.push(hash);
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // The chunk references are stored as a list of SHA-1 hashes
        // Chunk the bytes into 40-character strings
        let chunk_refs = bytes
            .chunks_exact(40)
            .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
            .collect();

        Ok(Self { chunk_refs })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        // Concatenate the chunk references into a single byte array
        self.chunk_refs
            .iter()
            .flat_map(|hash| hash.as_bytes().to_vec())
            .collect()
    }

    pub fn data_len(&self) -> usize {
        let store = Store::new().unwrap();
        self.chunk_refs
            .iter()
            .map(|hash| {
                let data = store.get_object(hash, ObjectType::Chunk).unwrap();
                Chunk::from_bytes(&data).unwrap().metadata().number_of_bytes
            })
            .sum()
    }

    pub fn data(&self) -> Vec<u8> {
        let store = Store::new().unwrap();
        self.chunk_refs
            .iter()
            .map(|hash| {
                let data = store.get_object(hash, ObjectType::Chunk).unwrap();
                Chunk::from_bytes(&data).unwrap().data().to_vec()
            })
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tile() {
        let refs = vec![
            "0123456789012345678901234567890123456789".to_string(),
            "9876543210987654321098765432109876543210".to_string(),
        ];
        let tile = Tile::new(refs.clone());
        assert_eq!(tile.chunk_refs(), refs);
    }

    #[test]
    #[should_panic]
    fn test_new_tile_invalid_hash() {
        // Invalid length
        let refs = vec!["123".to_string()];
        Tile::new(refs);
    }

    #[test]
    #[should_panic]
    fn test_new_tile_invalid_chars() {
        // Non-hex characters
        let refs = vec!["0123456789012345678901234567890123456xyz".to_string()];
        Tile::new(refs);
    }

    #[test]
    fn test_add_chunk_ref() {
        let mut tile = Tile::new(vec![]);
        let hash = "0123456789012345678901234567890123456789".to_string();
        tile.add_chunk_ref(hash.clone());
        assert_eq!(tile.chunk_refs(), vec![hash]);
    }

    #[test]
    fn test_empty_tile_serialization() {
        let tile = Tile::new(vec![]);
        let bytes = tile.to_bytes();
        let restored = Tile::from_bytes(&bytes).unwrap();
        assert_eq!(restored.chunk_refs(), tile.chunk_refs());
    }

    #[test]
    fn test_tile_serialization() {
        let refs = vec![
            "0123456789012345678901234567890123456789".to_string(),
            "9876543210987654321098765432109876543210".to_string(),
        ];
        let tile = Tile::new(refs);
        let bytes = tile.to_bytes();
        let restored = Tile::from_bytes(&bytes).unwrap();
        assert_eq!(restored.chunk_refs(), tile.chunk_refs());
    }

    #[test]
    #[should_panic]
    fn test_add_chunk_ref_invalid_hash() {
        let mut tile = Tile::new(vec![]);
        tile.add_chunk_ref("invalid".to_string());
    }

    #[test]
    fn test_serialization_large_tile() {
        let refs = (0..1000)
            .map(|_| "0123456789012345678901234567890123456789".to_string())
            .collect::<Vec<_>>();
        let tile = Tile::new(refs);
        let bytes = tile.to_bytes();
        let restored = Tile::from_bytes(&bytes).unwrap();
        assert_eq!(restored.chunk_refs(), tile.chunk_refs());
    }

    #[test]
    fn test_from_bytes_empty() {
        let bytes: Vec<u8> = vec![];
        let tile = Tile::from_bytes(&bytes).unwrap();
        assert!(tile.chunk_refs().is_empty());
    }
}
