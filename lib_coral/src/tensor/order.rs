use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Order {
    RowMajor,
    ColumnMajor,
}

impl Order {
    /// Generates the dimension ordering for the given number of dimensions
    /// For row-major: [0, 1, 2, ..., n-1]
    /// For column-major: [n-1, n-2, ..., 0]
    pub fn generate_ordering(&self, ndims: usize) -> Vec<usize> {
        match self {
            Order::RowMajor => (0..ndims).collect(),
            Order::ColumnMajor => (0..ndims).rev().collect(),
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Order::RowMajor => "RowMajor",
            Order::ColumnMajor => "ColumnMajor",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "RowMajor" => Some(Order::RowMajor),
            "ColumnMajor" => Some(Order::ColumnMajor),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        self.to_str().to_string()
    }

    pub fn from_le_bytes(bytes: [u8; 4]) -> Self {
        match u32::from_le_bytes(bytes) {
            0 => Order::RowMajor,
            1 => Order::ColumnMajor,
            _ => panic!("Invalid order"),
        }
    }

    pub fn to_le_bytes(&self) -> [u8; 4] {
        match self {
            Order::RowMajor => 0u32.to_le_bytes(),
            Order::ColumnMajor => 1u32.to_le_bytes(),
        }
    }
}
