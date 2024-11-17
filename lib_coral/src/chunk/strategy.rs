use crate::{chunk::chunk::ChunkMetadata, tensor::dtype::Dtype};

use super::chunk::Chunk;

/// Trait for a strategy to be used in chunking a tile.
pub trait Strategy {
    /// Chunk a tile.
    fn chunk<'a>(&self, data: &'a [u8], dtype: Dtype) -> Vec<Chunk<'a>>;
}

/// A strategy that chunks a tile into a single chunk.
///
/// This strategy is useful when the data is small enough to fit into a single chunk.
///
/// # Examples
///
/// ```
/// use coral::tensor::strategy::SingleChunk;
/// use coral::tensor::tile::Tile;
///
/// let data = vec![1, 2, 3, 4];
/// let tile = Tile::new(data);
/// let strategy = SingleChunk::new();
/// let chunks = strategy.chunk(tile.data());
/// assert_eq!(chunks.len(), 1);
/// assert_eq!(chunks[0].data().to_vec(), data);
/// ```
///
/// # Performance
///
/// This strategy is not suitable for large data, as it will load the entire data into memory.
/// For large data, consider using a different strategy.
///
/// # See also
///
/// - [`FixedChunk`]
/// - [`AdaptiveChunk`]
/// - [`AdaptiveChunkWithThreshold`]
/// - [`AdaptiveChunkWithMaxSize`]
/// - [`AdaptiveChunkWithMaxSizeAndThreshold`]
pub struct SingleChunk;

impl SingleChunk {
    /// Create a new instance of the strategy.
    pub fn new() -> Self {
        Self
    }
}

impl Strategy for SingleChunk {
    fn chunk<'a>(&self, data: &'a [u8], dtype: Dtype) -> Vec<Chunk<'a>> {
        let number_of_elements = data.len() / dtype.size();
        let metadata = ChunkMetadata::new(dtype, number_of_elements);
        vec![Chunk::new(metadata, data)]
    }
}

/// A strategy that chunks a tile into fixed-size chunks.
///
/// This strategy is useful when the data is too large to fit into a single chunk.
/// The data is split into fixed-size chunks, each with the same size. Respecting the dtype size so that no elements are split between chunks.
///
/// # Examples
///
/// ```
/// use coral::tensor::strategy::FixedChunk;
/// use coral::tensor::tile::Tile;
///
/// let data = vec![1, 2, 3, 4];
/// let tile = Tile::new(data);
/// let strategy = FixedChunk::new(2);
/// let chunks = strategy.chunk(tile.data());
/// assert_eq!(chunks.len(), 2);
/// assert_eq!(chunks[0].data().to_vec(), vec![1, 2]);
/// assert_eq!(chunks[1].data().to_vec(), vec![3, 4]);
/// ```
///
/// # Performance
///
/// This strategy is suitable for large data, as it will only load a single chunk into memory at a time.
///
/// # See also
///
/// - [`SingleChunk`]
pub struct FixedChunk {
    chunk_size: usize,
}

impl FixedChunk {
    /// Create a new instance of the strategy.
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl Strategy for FixedChunk {
    fn chunk<'a>(&self, data: &'a [u8], dtype: Dtype) -> Vec<Chunk<'a>> {
        let chunk_size = self.chunk_size * dtype.size();
        let mut chunks = Vec::new();
        for chunk_data in data.chunks(chunk_size) {
            let metadata = ChunkMetadata::new(dtype, chunk_data.len() / dtype.size());
            chunks.push(Chunk::new(metadata, chunk_data));
        }
        chunks
    }
}

/// A strategy that chunks a tile into chunks of varying sizes.
/// The strategy will attempt to split the data into chunks of approximately the same size.
/// The last chunk may be smaller if the data size is not divisible by the chunk size.
///
/// # Examples
/// ```
/// use coral::tensor::strategy::AdaptiveChunk;
/// use coral::tensor::tile::Tile;
///
/// let data = vec![1, 2, 3, 4];
/// let tile = Tile::new(data);
/// let strategy = AdaptiveChunk::new(2);
/// let chunks = strategy.chunk(tile.data());
/// assert_eq!(chunks.len(), 2);
/// assert_eq!(chunks[0].data().to_vec(), vec![1, 2]);
/// assert_eq!(chunks[1].data().to_vec(), vec![3, 4]);
/// ```
///
/// # Performance
///
/// This strategy is suitable for large data, as it will only load a single chunk into memory at a time.
///
/// # See also
///
/// - [`SingleChunk`]
/// - [`FixedChunk`]
pub struct AdaptiveChunk {
    chunk_size: usize,
}

impl AdaptiveChunk {
    /// Create a new instance of the strategy.
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl Strategy for AdaptiveChunk {
    fn chunk<'a>(&self, data: &'a [u8], dtype: Dtype) -> Vec<Chunk<'a>> {
        let chunk_size = self.chunk_size * dtype.size();
        let mut chunks = Vec::new();
        for chunk_data in data.chunks(chunk_size) {
            let metadata = ChunkMetadata::new(dtype, chunk_data.len() / dtype.size());
            chunks.push(Chunk::new(metadata, chunk_data));
        }
        chunks
    }
}
