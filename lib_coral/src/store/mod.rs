use anyhow::{anyhow, Result};
use sha1::{Digest, Sha1};
use std::fs;
use std::path::{Path, PathBuf};

/// Represents the different types of objects that can be stored
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Snapshot,
    Commit,
    Chunk,
}

impl ObjectType {
    fn as_str(&self) -> &'static str {
        match self {
            ObjectType::Snapshot => "snapshots",
            ObjectType::Commit => "commits",
            ObjectType::Chunk => "chunks",
        }
    }
}

/// Store manages objects in the .coral directory
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Create a new store, searching up from the current directory for .coral
    pub fn new() -> Result<Self> {
        let current_dir = std::env::current_dir()?;
        Self::find_store(&current_dir)
    }

    /// Create a new store from a specific path
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::find_store(path.as_ref())
    }

    /// Initialize a new .coral store in the given directory
    pub fn init(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().join(".coral");
        let objects = root.join("objects");

        // Create main directories
        fs::create_dir_all(&root)?;
        fs::create_dir_all(&objects)?;

        // Create subdirectories for each object type
        fs::create_dir_all(objects.join("snapshots"))?;
        fs::create_dir_all(objects.join("commits"))?;
        fs::create_dir_all(objects.join("chunks"))?;

        Ok(Store { root })
    }

    /// Find the .coral directory by walking up the directory tree
    fn find_store(start: &Path) -> Result<Self> {
        let mut current = start.to_path_buf();
        loop {
            let coral_dir = current.join(".coral");
            if coral_dir.is_dir() {
                return Ok(Store { root: coral_dir });
            }
            if !current.pop() {
                return Err(anyhow!("No .coral directory found"));
            }
        }
    }

    /// Get the path for an object given its hash and type
    fn object_path(&self, hash: &str, obj_type: ObjectType) -> PathBuf {
        let (prefix, rest) = hash.split_at(2);
        self.root
            .join("objects")
            .join(obj_type.as_str())
            .join(prefix)
            .join(rest)
    }

    /// Store an object with the given type
    pub fn store_object(&self, data: &[u8], obj_type: ObjectType) -> Result<String> {
        // Calculate hash
        let mut hasher = Sha1::new();
        hasher.update(data);
        let hash = hex::encode(hasher.finalize());

        // Create path
        let path = self.object_path(&hash, obj_type);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write file if it doesn't exist
        if !path.exists() {
            fs::write(&path, data)?;
        }

        Ok(hash)
    }

    /// Retrieve an object by its hash and type
    pub fn get_object(&self, hash: &str, obj_type: ObjectType) -> Result<Vec<u8>> {
        let path = self.object_path(hash, obj_type);
        if !path.exists() {
            return Err(anyhow!("Object not found: {}", hash));
        }
        Ok(fs::read(path)?)
    }

    /// Check if an object exists
    pub fn has_object(&self, hash: &str, obj_type: ObjectType) -> bool {
        self.object_path(hash, obj_type).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_store_init() -> Result<()> {
        let temp = tempdir()?;
        let _ = Store::init(temp.path())?;

        assert!(temp.path().join(".coral").is_dir());
        assert!(temp.path().join(".coral/objects/snapshots").is_dir());
        assert!(temp.path().join(".coral/objects/commits").is_dir());
        assert!(temp.path().join(".coral/objects/chunks").is_dir());

        Ok(())
    }

    #[test]
    fn test_store_object() -> Result<()> {
        let temp = tempdir()?;
        let store = Store::init(temp.path())?;

        let data = b"test data";
        let hash = store.store_object(data, ObjectType::Chunk)?;

        assert!(store.has_object(&hash, ObjectType::Chunk));
        assert_eq!(store.get_object(&hash, ObjectType::Chunk)?, data);

        Ok(())
    }
}
