/// Represents a reference in the coral system
/// Similar to Git refs, these point to specific snapshots
/// and can be either heads (branches) or tags
pub struct Ref {
    name: String,
    hash: [u8; 20],
    ref_type: RefType,
}

#[derive(Debug, PartialEq)]
pub enum RefType {
    Head, // For branches
    Tag,  // For tags
}

impl Ref {
    /// Create a new reference
    pub fn new(name: String, hash: [u8; 20], ref_type: RefType) -> Self {
        Self {
            name,
            hash,
            ref_type,
        }
    }

    /// Read a ref from the .coral directory
    pub fn find(name: &str, coral_dir: &Path) -> Result<Option<Self>> {
        // Try heads first
        let head_path = coral_dir.join("refs").join("heads").join(name);
        if head_path.exists() {
            let hash = fs::read(&head_path)?;
            if hash.len() == 20 {
                let mut hash_array = [0u8; 20];
                hash_array.copy_from_slice(&hash);
                return Ok(Some(Self::new(name.to_string(), hash_array, RefType::Head)));
            }
        }

        // Try tags if head wasn't found
        let tag_path = coral_dir.join("refs").join("tags").join(name);
        if tag_path.exists() {
            let hash = fs::read(&tag_path)?;
            if hash.len() == 20 {
                let mut hash_array = [0u8; 20];
                hash_array.copy_from_slice(&hash);
                return Ok(Some(Self::new(name.to_string(), hash_array, RefType::Tag)));
            }
        }

        Ok(None)
    }

    /// Write the ref to the appropriate location in the .coral directory
    pub fn save(&self, coral_dir: &Path) -> Result<()> {
        let base_path = match self.ref_type {
            RefType::Head => coral_dir.join("refs").join("heads"),
            RefType::Tag => coral_dir.join("refs").join("tags"),
        };

        fs::create_dir_all(&base_path)?;
        let ref_path = base_path.join(&self.name);
        fs::write(ref_path, &self.hash)?;

        Ok(())
    }

    /// List all refs in the .coral directory
    pub fn list_all(coral_dir: &Path) -> Result<Vec<Self>> {
        let mut refs = Vec::new();

        // List heads
        let heads_dir = coral_dir.join("refs").join("heads");
        if heads_dir.exists() {
            Self::read_refs_from_dir(&heads_dir, RefType::Head, &mut refs)?;
        }

        // List tags
        let tags_dir = coral_dir.join("refs").join("tags");
        if tags_dir.exists() {
            Self::read_refs_from_dir(&tags_dir, RefType::Tag, &mut refs)?;
        }

        Ok(refs)
    }

    /// Helper function to read refs from a directory
    fn read_refs_from_dir(dir: &Path, ref_type: RefType, refs: &mut Vec<Self>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let hash = fs::read(&path)?;
                    if hash.len() == 20 {
                        let mut hash_array = [0u8; 20];
                        hash_array.copy_from_slice(&hash);
                        refs.push(Self::new(name.to_string(), hash_array, ref_type));
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the name of the ref
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the hash this ref points to
    pub fn hash(&self) -> &[u8; 20] {
        &self.hash
    }

    /// Get the type of this ref
    pub fn ref_type(&self) -> &RefType {
        &self.ref_type
    }

    /// Get the snapshot this ref points to
    pub fn get_snapshot(&self, store: &Store) -> Result<Snapshot> {
        let snapshot_bytes = store.get_object(&self.hash, ObjectType::Snapshot)?;
        Snapshot::from_bytes(&snapshot_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_operations() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let coral_dir = temp_dir.path();

        // Create a test ref
        let hash = [1u8; 20];
        let test_ref = Ref::new("main".to_string(), hash, RefType::Head);

        // Save the ref
        test_ref.save(coral_dir)?;

        // Find the ref
        let found_ref = Ref::find("main", coral_dir)?.expect("Ref should exist");
        assert_eq!(found_ref.hash(), &hash);
        assert_eq!(found_ref.name(), "main");
        assert_eq!(found_ref.ref_type(), &RefType::Head);

        // List all refs
        let all_refs = Ref::list_all(coral_dir)?;
        assert_eq!(all_refs.len(), 1);
        assert_eq!(all_refs[0].name(), "main");

        Ok(())
    }
}
