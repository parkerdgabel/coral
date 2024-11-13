use std::{
    io::{Cursor, Read},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;

use byteorder::{LittleEndian, ReadBytesExt};

/// Represents a commit in the coral system
/// The on-disk format is:
/// - Header:
///     - Snapshot SHA-1 (20 bytes)
///     - Number of parents (4 bytes)
/// - Parent Commits:
///     - For each parent:
///         - Parent SHA-1 (20 bytes)
/// - Metadata:
///     - Author length (4 bytes)
///     - Author (variable length)
///     - Email length (4 bytes)
///     - Email (variable length)
///     - Message length (4 bytes)
///     - Message (variable length)
///     - Timestamp (8 bytes)
pub struct Commit {
    snapshot_hash: [u8; 20],
    parent_hashes: Vec<[u8; 20]>,
    author: String,
    email: String,
    message: String,
    timestamp: u64,
}

impl Commit {
    pub fn new(
        snapshot_hash: [u8; 20],
        parent_hashes: Vec<[u8; 20]>,
        author: String,
        email: String,
        message: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            snapshot_hash,
            parent_hashes,
            author,
            email,
            message,
            timestamp,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        // Read snapshot hash
        let mut snapshot_hash = [0u8; 20];
        cursor.read_exact(&mut snapshot_hash)?;

        // Read parent commits
        let num_parents = cursor.read_u32::<LittleEndian>()?;
        let mut parent_hashes = Vec::with_capacity(num_parents as usize);
        for _ in 0..num_parents {
            let mut parent_hash = [0u8; 20];
            cursor.read_exact(&mut parent_hash)?;
            parent_hashes.push(parent_hash);
        }

        // Read author
        let author_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut author_bytes = vec![0u8; author_len];
        cursor.read_exact(&mut author_bytes)?;
        let author = String::from_utf8(author_bytes)?;

        // Read email
        let email_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut email_bytes = vec![0u8; email_len];
        cursor.read_exact(&mut email_bytes)?;
        let email = String::from_utf8(email_bytes)?;

        // Read message
        let message_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut message_bytes = vec![0u8; message_len];
        cursor.read_exact(&mut message_bytes)?;
        let message = String::from_utf8(message_bytes)?;

        // Read timestamp
        let timestamp = cursor.read_u64::<LittleEndian>()?;

        Ok(Self {
            snapshot_hash,
            parent_hashes,
            author,
            email,
            message,
            timestamp,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write snapshot hash
        bytes.extend_from_slice(&self.snapshot_hash);

        // Write number of parents and parent hashes
        bytes.extend_from_slice(&(self.parent_hashes.len() as u32).to_le_bytes());
        for parent_hash in &self.parent_hashes {
            bytes.extend_from_slice(parent_hash);
        }

        // Write author
        bytes.extend_from_slice(&(self.author.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.author.as_bytes());

        // Write email
        bytes.extend_from_slice(&(self.email.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.email.as_bytes());

        // Write message
        bytes.extend_from_slice(&(self.message.len() as u32).to_le_bytes());
        bytes.extend_from_slice(self.message.as_bytes());

        // Write timestamp
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_serialization() {
        let snapshot_hash = [1u8; 20];
        let parent_hash = [2u8; 20];
        let commit = Commit::new(
            snapshot_hash,
            vec![parent_hash],
            "John Doe".to_string(),
            "john@example.com".to_string(),
            "Initial commit".to_string(),
        );

        let bytes = commit.to_bytes();
        let decoded = Commit::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.snapshot_hash, snapshot_hash);
        assert_eq!(decoded.parent_hashes, vec![parent_hash]);
        assert_eq!(decoded.author, "John Doe");
        assert_eq!(decoded.email, "john@example.com");
        assert_eq!(decoded.message, "Initial commit");
        assert_eq!(decoded.timestamp, commit.timestamp);
    }
}
