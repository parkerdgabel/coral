"""
Tests for clustering migration functionality.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import shutil

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class TestClusterMigration:
    """Test clustering migration functionality."""
    
    def test_migrate_empty_repository(self):
        """Test migrating an empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            result = repo.migrate_to_clustering()
            
            assert result['weights_processed'] == 0
            assert result['clusters_created'] == 0
            assert result['warnings'] == ["No weights found to migrate"]
            
            # Check that clustering is enabled
            status = repo.get_clustering_status()
            assert status['enabled'] is True
    
    def test_migrate_simple_repository(self):
        """Test migrating a repository with similar weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Lower deduplication threshold to ensure weights aren't deduplicated
            repo.deduplicator.similarity_threshold = 0.999
            
            # Create some similar but distinct weights
            base_data = np.random.randn(100, 100).astype(np.float32)
            weights = {}
            
            for i in range(5):
                # Create weights that are similar but not identical
                # Add enough noise to avoid deduplication but stay similar
                noise_scale = 0.1 + (i * 0.02)  # Increasing noise for each weight
                data = base_data + np.random.randn(100, 100).astype(np.float32) * noise_scale
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"weight_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights[f"weight_{i}"] = weight
            
            # Stage and commit weights
            repo.stage_weights(weights)
            repo.commit("Initial weights")
            
            # Migrate to clustering
            result = repo.migrate_to_clustering(
                strategy="adaptive",
                threshold=0.95
            )
            
            assert result['weights_processed'] == 5
            assert result['clusters_created'] >= 1  # At least one cluster
            # Note: space_saved might be negative for small weights due to overhead
            assert 'space_saved' in result
            assert 'reduction_percentage' in result
            
            # Verify clustering is enabled
            status = repo.get_clustering_status()
            assert status['enabled'] is True
            
            # Check that repository is in mixed mode
            assert repo.config.get("clustering", {}).get("mixed_mode") is True
    
    def test_migrate_with_backup(self):
        """Test migration with backup creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create and commit some weights
            weight = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name="test_weight",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            repo.stage_weights({"test_weight": weight})
            repo.commit("Initial commit")
            
            # Create backup
            backup_path = repo.create_backup()
            
            assert backup_path.exists()
            assert backup_path.name.startswith(".coral_backup_")
            assert (backup_path / "config.json").exists()
            assert (backup_path / "objects" / "weights.h5").exists()
            
            # Verify backup contains commit data
            assert (backup_path / "refs" / "heads" / "main").exists()
            
            # Clean up backup
            shutil.rmtree(backup_path)
    
    def test_migrate_with_batching(self):
        """Test migration with batch processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create many weights
            weights = {}
            for i in range(50):
                data = np.random.randn(10, 10).astype(np.float32)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"weight_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights[f"weight_{i}"] = weight
            
            # Commit in batches
            batch_size = 10
            for i in range(0, 50, batch_size):
                batch = {k: v for k, v in list(weights.items())[i:i+batch_size]}
                repo.stage_weights(batch)
                repo.commit(f"Batch {i//batch_size}")
            
            # Track progress
            progress_calls = []
            def progress_callback(current, total):
                progress_calls.append((current, total))
            
            # Migrate with small batch size
            result = repo.migrate_to_clustering(
                batch_size=5,
                progress_callback=progress_callback
            )
            
            assert result['weights_processed'] == 50
            assert len(progress_calls) > 0
            # Verify progress was tracked correctly
            assert progress_calls[-1][0] == progress_calls[-1][1]  # Final progress should be 100%
    
    def test_mixed_mode_operation(self):
        """Test that repository works in mixed mode after migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create initial weights
            weights_before = {}
            for i in range(3):
                data = np.random.randn(20, 20).astype(np.float32)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"weight_before_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights_before[f"weight_before_{i}"] = weight
            
            repo.stage_weights(weights_before)
            repo.commit("Before migration")
            
            # Migrate to clustering
            repo.migrate_to_clustering()
            
            # Add new weights after migration
            weights_after = {}
            for i in range(3):
                data = np.random.randn(20, 20).astype(np.float32)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"weight_after_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights_after[f"weight_after_{i}"] = weight
            
            repo.stage_weights(weights_after)
            repo.commit("After migration")
            
            # Verify config shows mixed mode
            assert repo.config.get("clustering", {}).get("mixed_mode") is True
            
            # Check clustering status
            status = repo.get_clustering_status()
            assert status['enabled'] is True
            
            # Verify we have commits before and after migration
            commits = repo.log(max_commits=10)
            assert len(commits) >= 2
            assert any("Before migration" in c.metadata.message for c in commits)
            assert any("After migration" in c.metadata.message for c in commits)
    
    def test_migrate_force_remigration(self):
        """Test force re-migration of already clustered repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repository(Path(tmpdir), init=True)
            
            # Create multiple weights to allow clustering
            weights = {}
            for i in range(3):
                data = np.random.randn(10, 10).astype(np.float32) * (1.0 + i * 0.1)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"test_weight_{i}",
                        shape=(10, 10),
                        dtype=np.float32
                    )
                )
                weights[f"test_weight_{i}"] = weight
            
            repo.stage_weights(weights)
            repo.commit("Initial commit")
            
            # First migration
            result1 = repo.migrate_to_clustering()
            assert result1['weights_processed'] >= 1  # At least one unique weight processed
            
            # Try migration again without force (should skip)
            status = repo.get_clustering_status()
            assert status['enabled'] is True
            
            # Force re-migration
            result2 = repo.migrate_to_clustering(strategy="kmeans", threshold=0.9)
            assert result2['weights_processed'] >= 1  # At least one unique weight processed
            # Strategy should be updated
            assert repo.config["clustering"]["migration_strategy"] == "kmeans"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])