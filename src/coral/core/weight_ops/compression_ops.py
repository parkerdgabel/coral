"""Compression operations for weight computation graphs."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from .base import OpType, WeightOp, validate_array


class SVDOp(WeightOp):
    """Low-rank approximation using Singular Value Decomposition (SVD).
    
    Decomposes a matrix as U @ S @ V^T, optionally keeping only the top-k
    singular values for compression.
    """
    
    def __init__(
        self, 
        U: np.ndarray, 
        S: np.ndarray, 
        V: np.ndarray,
        rank: Optional[int] = None,
        original_shape: Optional[Tuple[int, ...]] = None
    ):
        """Initialize SVD operation.
        
        Args:
            U: Left singular vectors
            S: Singular values (1D array)
            V: Right singular vectors 
            rank: Number of components to keep (None = keep all)
            original_shape: Original tensor shape if it was reshaped for SVD
            
        Raises:
            ValueError: If matrices have incompatible shapes
        """
        validate_array(U)
        validate_array(S)
        validate_array(V)
        
        if S.ndim != 1:
            raise ValueError(f"S must be 1D array, got shape {S.shape}")
        
        # Truncate if rank specified
        if rank is not None:
            if rank > len(S):
                raise ValueError(f"Rank {rank} exceeds number of singular values {len(S)}")
            U = U[:, :rank]
            S = S[:rank]
            V = V[:rank, :]
        
        # Validate shapes
        if U.shape[1] != len(S) or V.shape[0] != len(S):
            raise ValueError(
                f"Incompatible shapes: U={U.shape}, S={S.shape}, V={V.shape}"
            )
        
        self.U = U.copy()
        self.S = S.copy()
        self.V = V.copy()
        self.rank = rank or len(S)
        self.original_shape = original_shape
        
        # Compute output shape
        if original_shape:
            self._output_shape = original_shape
        else:
            self._output_shape = (U.shape[0], V.shape[1])
        
        self._output_dtype = np.result_type(U.dtype, V.dtype)
    
    def forward(self) -> np.ndarray:
        """Reconstruct the weight matrix from SVD components."""
        # Efficient reconstruction: (U @ diag(S)) @ V
        reconstructed = (self.U * self.S) @ self.V
        
        # Reshape to original if needed
        if self.original_shape:
            reconstructed = reconstructed.reshape(self.original_shape)
        
        return reconstructed
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of SVD components."""
        return self.U.nbytes + self.S.nbytes + self.V.nbytes
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": "SVD",
            "U": self.U.tolist(),
            "S": self.S.tolist(),
            "V": self.V.tolist(),
            "rank": self.rank,
            "original_shape": self.original_shape,
            "U_shape": self.U.shape,
            "V_shape": self.V.shape,
            "dtype": str(self._output_dtype)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "SVDOp":
        """Deserialize from stored representation."""
        U = np.array(data["U"], dtype=data["dtype"])
        S = np.array(data["S"], dtype=data["dtype"])
        V = np.array(data["V"], dtype=data["dtype"])
        return cls(U, S, V, data["rank"], data.get("original_shape"))


class SparseOp(WeightOp):
    """Sparse matrix storage operation.
    
    Supports multiple sparse formats: CSR, COO, CSC.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, sparse.spmatrix],
        indices: Optional[np.ndarray] = None,
        indptr: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, ...]] = None,
        format: str = "csr"
    ):
        """Initialize sparse operation.
        
        Args:
            data: Values array or full sparse matrix
            indices: Column indices (CSR/CSC) or coordinates (COO)
            indptr: Row pointers (CSR) or column pointers (CSC)
            shape: Shape of the sparse matrix
            format: Sparse format ('csr', 'coo', 'csc')
        """
        if format not in ["csr", "coo", "csc"]:
            raise ValueError(f"Unknown sparse format: {format}")
        
        self.format = format
        
        if isinstance(data, sparse.spmatrix):
            # Convert to desired format
            if format == "csr":
                matrix = data.tocsr()
                self.data = matrix.data
                self.indices = matrix.indices
                self.indptr = matrix.indptr
            elif format == "coo":
                matrix = data.tocoo()
                self.data = matrix.data
                self.row = matrix.row
                self.col = matrix.col
            elif format == "csc":
                matrix = data.tocsc()
                self.data = matrix.data
                self.indices = matrix.indices
                self.indptr = matrix.indptr
            self._output_shape = matrix.shape
        else:
            # Construct from arrays
            validate_array(data)
            self.data = data.copy()
            
            if format in ["csr", "csc"]:
                if indices is None or indptr is None:
                    raise ValueError(f"{format} format requires indices and indptr")
                self.indices = indices.copy()
                self.indptr = indptr.copy()
            elif format == "coo":
                if indices is None or len(indices) != 2:
                    raise ValueError("COO format requires (row, col) indices")
                self.row = indices[0].copy()
                self.col = indices[1].copy()
            
            if shape is None:
                raise ValueError("Shape must be provided when constructing from arrays")
            self._output_shape = shape
        
        self._output_dtype = self.data.dtype
    
    def forward(self) -> np.ndarray:
        """Convert sparse matrix to dense array."""
        if self.format == "csr":
            matrix = sparse.csr_matrix(
                (self.data, self.indices, self.indptr),
                shape=self._output_shape
            )
        elif self.format == "coo":
            matrix = sparse.coo_matrix(
                (self.data, (self.row, self.col)),
                shape=self._output_shape
            )
        elif self.format == "csc":
            matrix = sparse.csc_matrix(
                (self.data, self.indices, self.indptr),
                shape=self._output_shape
            )
        
        return matrix.toarray()
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of sparse representation."""
        memory = self.data.nbytes
        
        if self.format in ["csr", "csc"]:
            memory += self.indices.nbytes + self.indptr.nbytes
        elif self.format == "coo":
            memory += self.row.nbytes + self.col.nbytes
        
        return memory
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        result = {
            "type": "SPARSE",
            "format": self.format,
            "data": self.data.tolist(),
            "shape": self._output_shape,
            "dtype": str(self._output_dtype)
        }
        
        if self.format in ["csr", "csc"]:
            result["indices"] = self.indices.tolist()
            result["indptr"] = self.indptr.tolist()
        elif self.format == "coo":
            result["row"] = self.row.tolist()
            result["col"] = self.col.tolist()
        
        return result
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "SparseOp":
        """Deserialize from stored representation."""
        values = np.array(data["data"], dtype=data["dtype"])
        format = data["format"]
        
        if format in ["csr", "csc"]:
            indices = np.array(data["indices"])
            indptr = np.array(data["indptr"])
            return cls(values, indices, indptr, tuple(data["shape"]), format)
        elif format == "coo":
            row = np.array(data["row"])
            col = np.array(data["col"])
            indices = np.array([row, col])
            return cls(values, indices, shape=tuple(data["shape"]), format=format)


class QuantizeOp(WeightOp):
    """Quantization operation for reducing weight precision."""
    
    def __init__(
        self,
        quantized_data: np.ndarray,
        scale: float,
        zero_point: float,
        bits: int,
        symmetric: bool = True,
        original_dtype: Optional[np.dtype] = None
    ):
        """Initialize quantization operation.
        
        Args:
            quantized_data: Quantized weight data
            scale: Quantization scale factor
            zero_point: Quantization zero point
            bits: Number of bits used for quantization
            symmetric: Whether symmetric quantization was used
            original_dtype: Original data type before quantization
        """
        validate_array(quantized_data)
        
        if bits not in [2, 4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {bits}")
        
        self.quantized_data = quantized_data.copy()
        self.scale = float(scale)
        self.zero_point = float(zero_point)
        self.bits = bits
        self.symmetric = symmetric
        self.original_dtype = original_dtype or np.float32
        
        self._output_shape = quantized_data.shape
        self._output_dtype = np.dtype(self.original_dtype)
    
    def forward(self) -> np.ndarray:
        """Dequantize to reconstruct weight tensor."""
        if self.symmetric:
            # Symmetric dequantization
            dequantized = self.quantized_data.astype(np.float32) * self.scale
        else:
            # Asymmetric dequantization
            dequantized = (self.quantized_data.astype(np.float32) - self.zero_point) * self.scale
        
        return dequantized.astype(self.original_dtype)
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of quantized data."""
        # Quantized data + metadata overhead
        return self.quantized_data.nbytes + 16  # scale + zero_point
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": "QUANTIZE",
            "quantized_data": self.quantized_data.tolist(),
            "scale": self.scale,
            "zero_point": self.zero_point,
            "bits": self.bits,
            "symmetric": self.symmetric,
            "original_dtype": str(self.original_dtype),
            "shape": self._output_shape,
            "quantized_dtype": str(self.quantized_data.dtype)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "QuantizeOp":
        """Deserialize from stored representation."""
        quantized = np.array(data["quantized_data"], dtype=data["quantized_dtype"])
        return cls(
            quantized,
            data["scale"],
            data["zero_point"],
            data["bits"],
            data["symmetric"],
            np.dtype(data["original_dtype"])
        )


class PQOp(WeightOp):
    """Product Quantization operation for vector compression."""
    
    def __init__(
        self,
        indices: np.ndarray,
        codebooks: List[np.ndarray],
        residual: Optional[np.ndarray] = None,
        original_shape: Optional[Tuple[int, ...]] = None
    ):
        """Initialize PQ operation.
        
        Args:
            indices: Subvector indices for each codebook
            codebooks: List of codebook matrices
            residual: Optional residual for lossless reconstruction
            original_shape: Original tensor shape before flattening
        """
        validate_array(indices)
        
        if len(codebooks) == 0:
            raise ValueError("At least one codebook required")
        
        if len(indices) != len(codebooks):
            raise ValueError(
                f"Number of indices ({len(indices)}) must match "
                f"number of codebooks ({len(codebooks)})"
            )
        
        self.indices = indices.copy()
        self.codebooks = [cb.copy() for cb in codebooks]
        self.residual = residual.copy() if residual is not None else None
        self.original_shape = original_shape
        
        # Calculate output shape
        total_dim = sum(cb.shape[1] for cb in codebooks)
        if original_shape:
            self._output_shape = original_shape
        else:
            self._output_shape = (total_dim,)
        
        self._output_dtype = codebooks[0].dtype
    
    def forward(self) -> np.ndarray:
        """Decode using product quantization."""
        # Reconstruct by concatenating selected codewords
        reconstructed = []
        for idx, codebook in zip(self.indices, self.codebooks):
            if idx >= len(codebook):
                raise ValueError(f"Index {idx} out of range for codebook with {len(codebook)} entries")
            reconstructed.append(codebook[idx])
        
        reconstructed = np.concatenate(reconstructed)
        
        # Add residual if present
        if self.residual is not None:
            reconstructed = reconstructed + self.residual
        
        # Reshape to original if needed
        if self.original_shape:
            reconstructed = reconstructed.reshape(self.original_shape)
        
        return reconstructed
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return output shape without computing."""
        return self._output_shape
    
    def get_output_dtype(self) -> np.dtype:
        """Return output dtype without computing."""
        return self._output_dtype
    
    def get_memory_usage(self) -> int:
        """Return memory usage of PQ representation."""
        memory = self.indices.nbytes
        memory += sum(cb.nbytes for cb in self.codebooks)
        if self.residual is not None:
            memory += self.residual.nbytes
        return memory
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize operation for storage."""
        return {
            "type": "PQ",
            "indices": self.indices.tolist(),
            "codebooks": [cb.tolist() for cb in self.codebooks],
            "codebook_shapes": [cb.shape for cb in self.codebooks],
            "residual": self.residual.tolist() if self.residual is not None else None,
            "original_shape": self.original_shape,
            "dtype": str(self._output_dtype)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "PQOp":
        """Deserialize from stored representation."""
        indices = np.array(data["indices"])
        codebooks = [
            np.array(cb, dtype=data["dtype"]).reshape(shape)
            for cb, shape in zip(data["codebooks"], data["codebook_shapes"])
        ]
        residual = np.array(data["residual"]) if data["residual"] else None
        return cls(indices, codebooks, residual, data.get("original_shape"))