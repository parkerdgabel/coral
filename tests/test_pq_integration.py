"""Integration test for Product Quantization with HDF5 storage."""

import numpy as np
import tempfile
import os

from coral.storage.hdf5_store import HDF5Store
from coral.delta.product_quantization import PQCodebook, PQConfig, train_codebooks, encode_vector, decode_vector
from coral.delta.delta_encoder import Delta, DeltaType, DeltaEncoder
from coral.core.weight_tensor import WeightMetadata, WeightTensor


def test_pq_integration():
    """Test full PQ workflow: training, encoding, storage, and reconstruction."""
    
    # Create temporary storage
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name
    
    try:
        store = HDF5Store(temp_path)
        
        # Generate sample data - 100 vectors of dimension 32
        np.random.seed(42)
        vectors = np.random.randn(100, 32).astype(np.float32)
        
        # Train PQ codebook
        config = PQConfig(
            num_subvectors=8,
            bits_per_subvector=8,
            use_residual=True
        )
        
        codebook = train_codebooks(vectors, config)
        
        # Store the codebook
        codebook_id = "integration_test_cb"
        store.store_pq_codebook(codebook_id, codebook)
        
        # Create a reference weight
        reference_data = np.random.randn(32).astype(np.float32)
        reference = WeightTensor(
            data=reference_data,
            metadata=WeightMetadata(
                name="reference",
                shape=(32,),
                dtype=np.float32
            )
        )
        ref_hash = store.store(reference)
        
        # Create a similar weight to encode
        similar_data = reference_data + 0.1 * np.random.randn(32).astype(np.float32)
        
        # Encode with PQ
        indices, residual = encode_vector(similar_data - reference_data, codebook, config)
        
        # Create PQ delta with proper metadata
        # Store indices and residual in a single array as the encoder expects
        if residual is not None:
            # Pack indices and residual together
            indices_uint32 = indices.astype(np.uint32)
            residual_float32 = residual.astype(np.float32)
            encoded_data = np.concatenate([
                indices_uint32.view(np.uint8),
                residual_float32.view(np.uint8)
            ])
        else:
            indices_uint32 = indices.astype(np.uint32)
            encoded_data = indices_uint32.view(np.uint8)
        
        delta = Delta(
            delta_type=DeltaType.PQ_LOSSLESS,
            data=encoded_data,
            metadata={
                "codebook_id": codebook_id,
                "num_indices": len(indices),
                "indices_dtype": "uint32",
                "has_residual": residual is not None,
                "residual_dtype": "float32" if residual is not None else None,
                "original_shape": (32,),
                "num_subvectors": config.num_subvectors,
                "pq_config": {
                    "num_subvectors": config.num_subvectors,
                    "bits_per_subvector": config.bits_per_subvector,
                }
            },
            original_shape=(32,),
            original_dtype=np.float32,
            reference_hash=ref_hash,
            compression_ratio=0.85
        )
        
        # Store the delta
        delta_hash = "integration_delta_001"
        store.store_delta(delta, delta_hash)
        
        # Now reconstruct: Load everything back
        loaded_reference = store.load(ref_hash)
        loaded_delta = store.load_delta(delta_hash)
        loaded_codebook = store.load_pq_codebook(codebook_id)
        
        assert loaded_reference is not None
        assert loaded_delta is not None
        assert loaded_codebook is not None
        
        # For now, we need to add the codebook to the delta metadata
        # In a real implementation, the Repository class would handle this
        loaded_delta.metadata["codebook"] = loaded_codebook.to_dict()
        
        # Reconstruct the weight using the delta encoder
        encoder = DeltaEncoder()
        reconstructed = encoder.decode_delta(loaded_delta, loaded_reference)
        
        # The reconstruction won't be perfect due to PQ quantization,
        # but should be close
        reconstruction_error = np.mean(np.abs(reconstructed.data - similar_data))
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        
        # For lossless mode with residual, error should be very small
        assert reconstruction_error < 1e-5
        
        # Test garbage collection
        # Create an orphaned codebook
        orphaned_cb = PQCodebook(
            subvector_size=4,
            codebooks=[np.random.randn(16, 4).astype(np.float32)],
            version="1.0"
        )
        store.store_pq_codebook("orphaned_cb", orphaned_cb)
        
        # Run GC
        deleted = store.gc()
        print(f"Garbage collection results: {deleted}")
        
        # Should have deleted the orphaned codebook
        assert deleted["pq_codebooks"] == 1
        assert store.has_pq_codebook(codebook_id)  # Used one should remain
        assert not store.has_pq_codebook("orphaned_cb")  # Orphaned should be gone
        
        # Get final storage info
        storage_info = store.get_storage_info()
        pq_info = store.get_pq_storage_info()
        
        print(f"Storage info: {storage_info}")
        print(f"PQ info: {pq_info}")
        
        store.close()
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_pq_integration()
    print("Integration test passed!")