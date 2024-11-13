use std::{borrow::Cow, collections::HashMap, io::Read};

use anyhow::Result;

use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};

use super::{dtype::Dtype, order::Order, tile::Tile};

pub trait View {
    /// The `Dtype` of the tensor
    fn dtype(&self) -> Dtype;
    /// The shape of the tensor
    fn shape(&self) -> &[usize];
    /// The data of the tensor
    fn data(&self) -> Cow<[u8]>;
    /// The length of the data, in bytes.
    /// This is the length of the data returned by `data()` which excludes any of the Tensor metadata.
    /// This is necessary as this might be faster to get than `data().len()`
    /// for instance for tensors residing in GPU.
    fn data_len(&self) -> usize;
    // /// The order of the tiles in the tensor in decreasing order of dimension significance
    // fn tile_order(&self) -> Order;
    /// The order of the cells in the tensor
    fn cell_order(&self) -> Order;
    /// The extent of the tiles in the tensor
    fn tile_extent(&self) -> &[usize];
}

#[derive(Debug)]
struct TileBlock {
    offset: u64,
    size: u32,
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct TileCoord {
    coords: Vec<usize>,
}

/// Reperesents a view into a tensor in a file
/// The tensor is stored in a file in the following format:
/// - Header:
///     - Dtype (4 bytes)
///     - Number of dimensions (4 bytes)
///     - Shape (8 bytes per dimension)
///     - Tile extent (8 bytes per dimension)
///     - Tile order (8 bytes per dimension)
///     - Cell order (8 bytes per dimension)
/// - Index:
///    - Number of tiles (4 bytes)
///    - Tile blocks
///       - Tile coordinates (8 bytes per dimension)
///       - Offset (8 bytes)
///       - Size (4 bytes)
/// - Data:
///    - The data itself
pub struct TensorView<'data> {
    dtype: Dtype,
    shape: Vec<usize>,
    cell_order: Order,
    tile_extent: Vec<usize>,
    tile_blocks: HashMap<TileCoord, TileBlock>,
    data: &'data [u8],
}

impl<'data> TensorView<'data> {
    pub fn new(
        dtype: Dtype,
        shape: Vec<usize>,
        cell_order: Order,
        tile_extent: Vec<usize>,
        tile_coords: Vec<Vec<usize>>,
        data: &'data [u8],
    ) -> Self {
        let mut tile_blocks = HashMap::new();
        let mut current_offset = 0u64;

        // Calculate tile size
        let tile_size = tile_extent.iter().product::<usize>() * dtype.size();

        // Create tile blocks from provided coordinates
        for coords in tile_coords {
            tile_blocks.insert(
                TileCoord { coords },
                TileBlock {
                    offset: current_offset,
                    size: tile_size as u32,
                },
            );
            current_offset += tile_size as u64;
        }

        Self {
            dtype,
            shape,
            cell_order,
            tile_extent,
            tile_blocks,
            data,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Write header
        bytes.extend_from_slice(&self.dtype.to_le_bytes());
        bytes.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for dim in &self.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        for extent in &self.tile_extent {
            bytes.extend_from_slice(&(*extent as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&self.cell_order.to_le_bytes());

        // Write index
        bytes.extend_from_slice(&(self.tile_blocks.len() as u32).to_le_bytes());
        for (coords, block) in &self.tile_blocks {
            for coord in &coords.coords {
                bytes.extend_from_slice(&(*coord as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&block.offset.to_le_bytes());
            bytes.extend_from_slice(&block.size.to_le_bytes());
        }

        // Write data
        bytes.extend_from_slice(self.data);

        bytes
    }

    pub fn from_bytes(bytes: &'data [u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        let dtype_val = cursor.read_u32::<LittleEndian>()?;
        let dtype = Dtype::from_le_bytes(dtype_val.to_le_bytes());
        let ndims = cursor.read_u32::<LittleEndian>()? as usize;

        let mut shape = Vec::with_capacity(ndims);
        let mut tile_extent = Vec::with_capacity(ndims);

        for _ in 0..ndims {
            shape.push(cursor.read_u64::<LittleEndian>()? as usize);
        }

        for _ in 0..ndims {
            tile_extent.push(cursor.read_u64::<LittleEndian>()? as usize);
        }

        // Read the Order enums (4 bytes each)
        let mut cell_order_bytes = [0u8; 4];
        cursor.read_exact(&mut cell_order_bytes)?;
        let cell_order = Order::from_le_bytes(cell_order_bytes);

        // Read tile index
        let num_tiles = cursor.read_u32::<LittleEndian>()?;
        let mut tile_blocks = HashMap::new();

        for _ in 0..num_tiles {
            let mut coords = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                coords.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let offset = cursor.read_u64::<LittleEndian>()?;
            let size = cursor.read_u32::<LittleEndian>()?;

            tile_blocks.insert(TileCoord { coords }, TileBlock { offset, size });
        }

        let data = &bytes[cursor.position() as usize..];

        Ok(TensorView {
            dtype,
            shape,
            cell_order,
            tile_extent,
            tile_blocks,
            data,
        })
    }

    fn tile_block(&self, coords: &[usize]) -> Option<&TileBlock> {
        self.tile_blocks.get(&TileCoord {
            coords: coords.to_vec(),
        })
    }

    pub fn get_tile(&self, coords: &[usize]) -> Option<Tile> {
        if let Some(block) = self.tile_block(coords) {
            let offset = block.offset as usize;
            let size = block.size as usize;
            Tile::from_bytes(&self.data[offset..offset + size]).ok()
        } else {
            None
        }
    }

    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.tile_blocks
            .keys()
            .map(move |coords| self.get_tile(&coords.coords).unwrap())
    }
}

impl<'data> View for TensorView<'data> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        unimplemented!("This is a view, it doesn't have the data");
    }

    fn data_len(&self) -> usize {
        let tiles = self.tiles().collect::<Vec<_>>();
        let mut data_len = 0;
        for tile in tiles {
            data_len += tile.data_len();
        }
        data_len
    }

    fn cell_order(&self) -> Order {
        self.cell_order
    }

    fn tile_extent(&self) -> &[usize] {
        &self.tile_extent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_creation() {
        let dtype = Dtype::F32;
        let shape = vec![2, 3];
        let cell_order = Order::RowMajor;
        let tile_extent = vec![1, 3];
        let tile_coords = vec![vec![0, 0], vec![1, 0]];
        let data = vec![0u8; 24];

        let view = TensorView::new(
            dtype,
            shape.clone(),
            cell_order,
            tile_extent.clone(),
            tile_coords,
            &data,
        );

        assert_eq!(view.dtype(), dtype);
        assert_eq!(view.shape(), shape);
        assert_eq!(view.cell_order(), cell_order);
        assert_eq!(view.tile_extent(), tile_extent);
    }

    #[test]
    fn test_tensor_view_from_bytes() {
        let mut bytes = Vec::new();

        // Write header
        bytes.extend_from_slice(&Dtype::F32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // ndims
        bytes.extend_from_slice(&2u64.to_le_bytes()); // shape[0]
        bytes.extend_from_slice(&3u64.to_le_bytes()); // shape[1]
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tile_extent[0]
        bytes.extend_from_slice(&3u64.to_le_bytes()); // tile_extent[1]
        bytes.extend_from_slice(&Order::RowMajor.to_le_bytes());

        // Write index
        bytes.extend_from_slice(&2u32.to_le_bytes()); // num_tiles
                                                      // Tile 1
        bytes.extend_from_slice(&0u64.to_le_bytes()); // coord[0]
        bytes.extend_from_slice(&0u64.to_le_bytes()); // coord[1]
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
        bytes.extend_from_slice(&12u32.to_le_bytes()); // size
                                                       // Tile 2
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&12u64.to_le_bytes());
        bytes.extend_from_slice(&12u32.to_le_bytes());

        // Data
        bytes.extend_from_slice(&[0u8; 24]);

        let view = TensorView::from_bytes(&bytes).unwrap();
        assert_eq!(view.dtype(), Dtype::F32);
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.cell_order(), Order::RowMajor);
        assert_eq!(view.tile_extent(), &[1, 3]);
    }

    #[test]
    #[should_panic(expected = "This is a view")]
    fn test_tensor_view_data_unimplemented() {
        let view = TensorView::new(
            Dtype::F32,
            vec![2, 2],
            Order::RowMajor,
            vec![1, 2],
            vec![vec![0, 0], vec![1, 0]],
            &[0u8; 16],
        );
        let _ = view.data();
    }

    #[test]
    fn test_tensor_view_empty_shape() {
        let dtype = Dtype::F32;
        let shape = vec![];
        let cell_order = Order::RowMajor;
        let tile_extent = vec![];
        let tile_coords = vec![vec![]];
        let data = vec![0u8; 4];

        let view = TensorView::new(
            dtype,
            shape.clone(),
            cell_order,
            tile_extent.clone(),
            tile_coords,
            &data,
        );

        assert_eq!(view.shape().len(), 0);
        assert_eq!(view.tile_extent().len(), 0);
    }

    #[test]
    fn test_tensor_view_single_element() {
        let dtype = Dtype::F32;
        let shape = vec![1];
        let cell_order = Order::RowMajor;
        let tile_extent = vec![1];
        let tile_coords = vec![vec![0]];
        let data = vec![0u8; 4];

        let view = TensorView::new(
            dtype,
            shape.clone(),
            cell_order,
            tile_extent.clone(),
            tile_coords,
            &data,
        );

        assert_eq!(view.shape(), &[1]);
        assert_eq!(view.tile_extent(), &[1]);
    }

    #[test]
    fn test_tensor_view_different_dtypes() {
        for dtype in [Dtype::F32, Dtype::F64, Dtype::I32, Dtype::I64].iter() {
            let shape = vec![2, 2];
            let cell_order = Order::RowMajor;
            let tile_extent = vec![1, 2];
            let tile_coords = vec![vec![0, 0], vec![1, 0]];
            let data = vec![0u8; 16 * dtype.size()];

            let view = TensorView::new(
                *dtype,
                shape.clone(),
                cell_order,
                tile_extent.clone(),
                tile_coords.clone(),
                &data,
            );

            assert_eq!(view.dtype(), *dtype);
        }
    }

    #[test]
    fn test_tensor_view_column_major() {
        let dtype = Dtype::F32;
        let shape = vec![2, 3];
        let cell_order = Order::ColumnMajor;
        let tile_extent = vec![1, 3];
        let tile_coords = vec![vec![0, 0], vec![1, 0]];
        let data = vec![0u8; 24];

        let view = TensorView::new(
            dtype,
            shape.clone(),
            cell_order,
            tile_extent.clone(),
            tile_coords,
            &data,
        );

        assert_eq!(view.cell_order(), Order::ColumnMajor);
    }

    #[test]
    #[should_panic(expected = "unimplemented")]
    fn test_tensor_view_data_len_unimplemented() {
        let view = TensorView::new(
            Dtype::F32,
            vec![2, 2],
            Order::RowMajor,
            vec![1, 2],
            vec![vec![0, 0], vec![1, 0]],
            &[0u8; 16],
        );
        let _ = view.data_len();
    }
}
