use anyhow::Result;
use chunk::strategy::{SingleChunk, Strategy};
use tensor::{order::Order, tensor::TensorView};

pub mod chunk;
pub mod commit;
pub mod config;
pub mod refs;
pub mod snapshot;
pub mod store;
pub mod tensor;

pub struct TileIterator<'a> {
    data: &'a [u8],
    shape: Vec<usize>,
    tile_extent: Vec<usize>,
    cell_order: Order,
    current_index: usize,
    total_tiles: usize,
    dtype_size: usize,
}

impl<'a> TileIterator<'a> {
    pub fn new(
        data: &'a [u8],
        shape: Vec<usize>,
        tile_extent: Vec<usize>,
        order: Order,
        dtype_size: usize,
    ) -> Self {
        let cell_order = order;
        let total_tiles = shape
            .iter()
            .zip(tile_extent.iter())
            .map(|(&s, &e)| (s + e - 1) / e)
            .product();

        Self {
            data,
            shape,
            tile_extent,
            cell_order,
            current_index: 0,
            total_tiles,
            dtype_size,
        }
    }

    fn get_tile_coords(&self, tile_index: usize) -> Vec<usize> {
        let mut coords = vec![0; self.shape.len()];
        let mut remaining = tile_index;

        let dims: Box<dyn Iterator<Item = usize>> = match self.cell_order {
            Order::RowMajor => Box::new((0..self.shape.len()).rev()),
            Order::ColumnMajor => Box::new(0..self.shape.len()),
        };

        for dim in dims {
            let tile_count = (self.shape[dim] + self.tile_extent[dim] - 1) / self.tile_extent[dim];
            coords[dim] = remaining % tile_count;
            remaining /= tile_count;
        }
        coords
    }

    fn extract_tile(&self, tile_coords: &[usize]) -> Vec<u8> {
        let mut tile = Vec::new();
        let ndims = self.shape.len();

        let mut starts = vec![0; ndims];
        let mut ends = vec![0; ndims];

        for dim in 0..ndims {
            starts[dim] = tile_coords[dim] * self.tile_extent[dim];
            ends[dim] = (starts[dim] + self.tile_extent[dim]).min(self.shape[dim]);
        }

        let mut indices = starts.clone();
        loop {
            let mut offset = 0;
            let mut multiplier = 1;

            match self.cell_order {
                Order::RowMajor => {
                    for dim in (0..ndims).rev() {
                        offset += indices[dim] * multiplier;
                        multiplier *= self.shape[dim];
                    }
                }
                Order::ColumnMajor => {
                    for dim in 0..ndims {
                        offset += indices[dim] * multiplier;
                        multiplier *= self.shape[dim];
                    }
                }
            }

            let data_start = offset * self.dtype_size;
            let data_end = data_start + self.dtype_size;

            if data_end <= self.data.len() {
                tile.extend_from_slice(&self.data[data_start..data_end]);
            }

            let mut dim = ndims - 1;
            loop {
                indices[dim] += 1;
                if indices[dim] < ends[dim] {
                    break;
                }
                indices[dim] = starts[dim];
                if dim == 0 {
                    return tile;
                }
                dim -= 1;
            }
        }
    }
}

impl<'a> Iterator for TileIterator<'a> {
    type Item = (Vec<usize>, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_tiles {
            return None;
        }

        let coords = self.get_tile_coords(self.current_index);
        let tile = self.extract_tile(&coords);
        self.current_index += 1;

        Some((coords, tile))
    }
}

fn tile_tensor(
    data: &[u8],
    shape: Vec<usize>,
    tile_extent: Vec<usize>,
    order: Order,
    dtype_size: usize,
) -> Vec<(Vec<usize>, Vec<u8>)> {
    let iter = TileIterator::new(data, shape, tile_extent, order, dtype_size);
    iter.collect()
}

/// SnapshotConfig is a configuration struct for creating a snapshot.
pub struct SnapshotConfig {
    chunk_strategy: Option<Box<dyn Strategy>>,
}

impl SnapshotConfig {
    pub fn new() -> Self {
        Self {
            chunk_strategy: None,
        }
    }

    pub fn with_chunk_strategy(strategy: Box<dyn Strategy>) -> Self {
        Self {
            chunk_strategy: Some(strategy),
        }
    }

    pub fn chunk_strategy(mut self, strategy: Box<dyn Strategy>) -> Self {
        self.chunk_strategy = Some(strategy);
        self
    }
}

pub fn snapshot<
    S: AsRef<str> + Ord + std::fmt::Display,
    V: tensor::tensor::View,
    I: IntoIterator<Item = (S, V)>,
>(
    data: I,
    snapshot_config: SnapshotConfig,
) -> Result<String> {
    let store = store::Store::new()?;
    let chunk_strategy = snapshot_config
        .chunk_strategy
        .unwrap_or(Box::new(SingleChunk::new()));
    let mut snapshot_writer = snapshot::SnapshotWriter::new();
    for (name, tensor) in data {
        let cell_order = tensor.cell_order();
        let data = tensor.data();
        let shape = tensor.shape().to_vec();
        let tile_extent = tensor.tile_extent().to_vec();
        let dtype_size = tensor.dtype().size();

        let tiles = tile_tensor(
            &data,
            shape.clone(),
            tile_extent.clone(),
            cell_order,
            dtype_size,
        );

        let mut tensor_writer =
            tensor::tensor::TensorWriter::new(tensor.dtype(), shape, cell_order, tile_extent);

        for (coords, tile_data) in tiles {
            let chunk = chunk_strategy.chunk(&tile_data, tensor.dtype());
            let chunk_refs = chunk
                .iter()
                .map(|chunk| store.store_object(&chunk.to_bytes(), store::ObjectType::Chunk))
                .collect::<Result<Vec<_>>>()?;
            let tile = tensor::tile::Tile::new(chunk_refs);
            tensor_writer.add_tile(coords, tile);
        }
        snapshot_writer.add_tensor(name.to_string(), tensor_writer);
    }
    let snapshot_bytes = snapshot_writer.to_bytes();
    let snapshot_hash = store.store_object(&snapshot_bytes, store::ObjectType::Snapshot)?;

    Ok(snapshot_hash)
}

pub fn load_snapshot(snapshot_hash: &str) -> Result<snapshot::Snapshot> {
    let store = store::Store::new()?;
    let snapshot_bytes = store.get_object(snapshot_hash, store::ObjectType::Snapshot)?;
    snapshot::Snapshot::from_bytes(snapshot_bytes)
}

pub fn get_tensor(snapshot_hash: &str, tensor_name: &str) -> Result<TensorView> {
    let snapshot = load_snapshot(snapshot_hash)?;
    snapshot
        .get_tensor(tensor_name)
        .map(|tensor| tensor.clone())
        .ok_or_else(|| anyhow::anyhow!("Tensor not found"))
}

pub fn commit(
    snapshot_hash: &str,
    parent_hashes: Vec<String>,
    author: &str,
    email: &str,
    message: &str,
) -> Result<String> {
    let store = store::Store::new()?;
    let commit = commit::Commit::new(
        {
            let mut hash = [0u8; 20];
            hash.copy_from_slice(snapshot_hash.as_bytes());
            hash
        },
        parent_hashes
            .iter()
            .map(|s| {
                let mut hash = [0u8; 20];
                hash.copy_from_slice(s.as_bytes());
                hash
            })
            .collect(),
        author.to_string(),
        email.to_string(),
        message.to_string(),
    );
    let commit_bytes = commit.to_bytes();
    let commit_hash = store.store_object(&commit_bytes, store::ObjectType::Commit)?;

    Ok(commit_hash)
}

pub fn get_commit(commit_hash: &str) -> Result<commit::Commit> {
    let store = store::Store::new()?;
    let commit_bytes = store.get_object(commit_hash, store::ObjectType::Commit)?;
    commit::Commit::from_bytes(&commit_bytes)
}

pub fn list_refs(coral_dir: &std::path::Path) -> Result<Vec<refs::Ref>> {
    refs::Ref::list_all(coral_dir)
}

pub fn create_ref(coral_dir: &std::path::Path, name: &str, hash: &str) -> Result<()> {
    let store = store::Store::new()?;
    let ref_bytes = store.get_object(hash, store::ObjectType::Commit)?;
    let ref_path = coral_dir.join("refs").join("heads").join(name);
    std::fs::write(ref_path, ref_bytes)?;

    Ok(())
}

pub fn get_config() -> Result<config::Config> {
    config::Config::load()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tiling_row_major() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let shape = vec![3, 3];
        let tile_extent = vec![2, 2];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 4);

        assert_eq!(tiles[0], (vec![0, 0], vec![1, 2, 4, 5]));
        assert_eq!(tiles[1], (vec![0, 1], vec![3, 6]));
        assert_eq!(tiles[2], (vec![1, 0], vec![7, 8]));
        assert_eq!(tiles[3], (vec![1, 1], vec![9]));
    }

    #[test]
    fn test_basic_tiling_column_major() {
        let data: Vec<u8> = vec![1, 4, 7, 2, 5, 8, 3, 6, 9];
        let shape = vec![3, 3];
        let tile_extent = vec![2, 2];
        let order = Order::ColumnMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 4);
        // Note: Column major tile coordinates are the same, but data layout differs
        assert_eq!(tiles[0].0, vec![0, 0]);
        assert_eq!(tiles[1].0, vec![1, 0]);
        assert_eq!(tiles[2].0, vec![0, 1]);
        assert_eq!(tiles[3].0, vec![1, 1]);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let shape = vec![0, 0];
        let tile_extent = vec![2, 2];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let data: Vec<u8> = vec![1];
        let shape = vec![1, 1];
        let tile_extent = vec![1, 1];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], (vec![0, 0], vec![1]));
    }

    #[test]
    fn test_larger_dtype() {
        let data: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        let shape = vec![3];
        let tile_extent = vec![2];
        let order = Order::RowMajor;
        let dtype_size = 4;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0], (vec![0], vec![1, 0, 0, 0, 2, 0, 0, 0]));
        assert_eq!(tiles[1], (vec![1], vec![3, 0, 0, 0]));
    }

    #[test]
    fn test_3d_array() {
        let data: Vec<u8> = (1..=27).collect();
        let shape = vec![3, 3, 3];
        let tile_extent = vec![2, 2, 2];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 8);
        // Check that each tile has coordinates with 3 dimensions
        for tile in tiles {
            assert_eq!(tile.0.len(), 3);
        }
    }

    #[test]
    fn test_tile_extent_larger_than_shape() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tile_extent = vec![5, 5];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], (vec![0, 0], vec![1, 2, 3, 4]));
    }

    #[test]
    #[should_panic]
    fn test_zero_tile_extent() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tile_extent = vec![0, 0];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let _tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
    }

    #[test]
    fn test_tile_extent_equals_shape() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tile_extent = vec![2, 3];
        let order = Order::RowMajor;
        let dtype_size = 1;

        let tiles = tile_tensor(&data, shape, tile_extent, order, dtype_size);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], (vec![0, 0], vec![1, 2, 3, 4, 5, 6]));
    }
}
