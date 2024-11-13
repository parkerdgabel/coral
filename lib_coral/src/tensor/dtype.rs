use serde::{Deserialize, Serialize};

/// The various available dtypes. They MUST be in increasing alignment order
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum Dtype {
    /// Boolan type
    BOOL,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
}

impl Dtype {
    /// Gives out the size (in bytes) of 1 element of this dtype.
    pub fn size(&self) -> usize {
        match self {
            Dtype::BOOL => 1,
            Dtype::U8 => 1,
            Dtype::I8 => 1,
            Dtype::I16 => 2,
            Dtype::U16 => 2,
            Dtype::I32 => 4,
            Dtype::U32 => 4,
            Dtype::I64 => 8,
            Dtype::U64 => 8,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
            Dtype::F32 => 4,
            Dtype::F64 => 8,
        }
    }

    pub fn from_le_bytes(bytes: [u8; 4]) -> Self {
        match u32::from_le_bytes(bytes) {
            0 => Dtype::BOOL,
            1 => Dtype::U8,
            2 => Dtype::U16,
            3 => Dtype::U32,
            4 => Dtype::U64,
            5 => Dtype::I8,
            6 => Dtype::I16,
            7 => Dtype::I32,
            8 => Dtype::I64,
            9 => Dtype::F32,
            10 => Dtype::F64,
            11 => Dtype::F16,
            12 => Dtype::BF16,
            _ => panic!("Invalid Dtype value"),
        }
    }

    pub fn to_le_bytes(&self) -> [u8; 4] {
        match self {
            Dtype::BOOL => 0u32.to_le_bytes(),
            Dtype::U8 => 1u32.to_le_bytes(),
            Dtype::U16 => 2u32.to_le_bytes(),
            Dtype::U32 => 3u32.to_le_bytes(),
            Dtype::U64 => 4u32.to_le_bytes(),
            Dtype::I8 => 5u32.to_le_bytes(),
            Dtype::I16 => 6u32.to_le_bytes(),
            Dtype::I32 => 7u32.to_le_bytes(),
            Dtype::I64 => 8u32.to_le_bytes(),
            Dtype::F32 => 9u32.to_le_bytes(),
            Dtype::F64 => 10u32.to_le_bytes(),
            Dtype::F16 => 11u32.to_le_bytes(),
            Dtype::BF16 => 12u32.to_le_bytes(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::BOOL.size(), 1);
        assert_eq!(Dtype::U8.size(), 1);
        assert_eq!(Dtype::I8.size(), 1);
        assert_eq!(Dtype::I16.size(), 2);
        assert_eq!(Dtype::U16.size(), 2);
        assert_eq!(Dtype::F16.size(), 2);
        assert_eq!(Dtype::BF16.size(), 2);
        assert_eq!(Dtype::I32.size(), 4);
        assert_eq!(Dtype::U32.size(), 4);
        assert_eq!(Dtype::F32.size(), 4);
        assert_eq!(Dtype::F64.size(), 8);
        assert_eq!(Dtype::I64.size(), 8);
        assert_eq!(Dtype::U64.size(), 8);
    }

    #[test]
    fn test_dtype_bytes_roundtrip() {
        let types = [
            Dtype::BOOL,
            Dtype::U8,
            Dtype::I8,
            Dtype::U16,
            Dtype::I16,
            Dtype::F16,
            Dtype::BF16,
            Dtype::U32,
            Dtype::I32,
            Dtype::F32,
            Dtype::U64,
            Dtype::I64,
            Dtype::F64,
        ];

        for dtype in types {
            let bytes = dtype.to_le_bytes();
            let recovered = Dtype::from_le_bytes(bytes);
            assert_eq!(dtype, recovered, "Roundtrip failed for {:?}", dtype);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Dtype value")]
    fn test_invalid_dtype_bytes() {
        Dtype::from_le_bytes(100u32.to_le_bytes());
    }
}
