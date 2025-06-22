"""
Tests for the coral-ml cluster migrate CLI command.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class TestCLIMigration:
    """Test the cluster migrate CLI command."""
    
    def test_migrate_help(self):
        """Test that migrate help is available."""
        result = subprocess.run(
            ["coral-ml", "cluster", "migrate", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Migrate existing repository to use clustering" in result.stdout
        assert "--strategy" in result.stdout
        assert "--threshold" in result.stdout
        assert "--backup" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--force" in result.stdout
    
    def test_migrate_dry_run(self):
        """Test dry run migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a repository with some weights
            repo = Repository(Path(tmpdir), init=True)
            
            # Add some weights
            weights = {}
            for i in range(3):
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
            
            repo.stage_weights(weights)
            repo.commit("Initial weights")
            
            # Run dry-run migration
            result = subprocess.run(
                ["coral-ml", "cluster", "migrate", "--dry-run"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Analyzing repository for migration (dry run)" in result.stdout
            assert "Total weights:" in result.stdout
            assert "Potential clusters:" in result.stdout
            assert "No changes made (dry run)" in result.stdout
    
    def test_migrate_basic(self):
        """Test basic migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a repository with similar weights
            repo = Repository(Path(tmpdir), init=True)
            
            base_data = np.random.randn(20, 20).astype(np.float32)
            weights = {}
            
            for i in range(5):
                # Create similar weights
                data = base_data + np.random.randn(20, 20).astype(np.float32) * 0.01
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"layer_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights[f"layer_{i}"] = weight
            
            repo.stage_weights(weights)
            repo.commit("Model checkpoint")
            
            # Run migration
            result = subprocess.run(
                ["coral-ml", "cluster", "migrate", "--threshold", "0.95"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Migrating repository to use clustering" in result.stdout
            assert "Migration complete:" in result.stdout
            assert "Weights processed:" in result.stdout
            assert "Clusters created:" in result.stdout
            assert "✓ Migration validation passed" in result.stdout
    
    def test_migrate_with_backup(self):
        """Test migration with backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple repository
            repo = Repository(Path(tmpdir), init=True)
            
            weight = WeightTensor(
                data=np.random.randn(10, 10).astype(np.float32),
                metadata=WeightMetadata(
                    name="test_weight",
                    shape=(10, 10),
                    dtype=np.float32
                )
            )
            repo.stage_weights({"test_weight": weight})
            repo.commit("Initial")
            
            # Run migration with backup
            result = subprocess.run(
                ["coral-ml", "cluster", "migrate", "--backup"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Creating backup..." in result.stdout
            assert "Backup created at:" in result.stdout
            
            # Check that backup was created
            coral_dir = Path(tmpdir) / ".coral"
            backup_dirs = list(Path(tmpdir).glob(".coral_backup_*"))
            assert len(backup_dirs) == 1
    
    def test_migrate_already_enabled(self):
        """Test migration when clustering is already enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repository and enable clustering
            repo = Repository(Path(tmpdir), init=True)
            repo.enable_clustering()
            
            # Try to migrate
            result = subprocess.run(
                ["coral-ml", "cluster", "migrate"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Clustering is already enabled" in result.stdout
            assert "Use --force to re-run migration" in result.stdout
    
    def test_migrate_force(self):
        """Test force migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repository with clustering enabled
            repo = Repository(Path(tmpdir), init=True)
            
            # Add a weight
            weight = WeightTensor(
                data=np.random.randn(5, 5).astype(np.float32),
                metadata=WeightMetadata(
                    name="weight",
                    shape=(5, 5),
                    dtype=np.float32
                )
            )
            repo.stage_weights({"weight": weight})
            repo.commit("Initial")
            
            # Enable clustering
            repo.enable_clustering()
            
            # Force re-migration with different strategy
            result = subprocess.run(
                ["coral-ml", "cluster", "migrate", "--force", "--strategy", "kmeans"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Migrating repository to use clustering" in result.stdout
            assert "Strategy: kmeans" in result.stdout
            assert "Migration complete:" in result.stdout
    
    def test_migrate_with_options(self):
        """Test migration with various options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repository
            repo = Repository(Path(tmpdir), init=True)
            
            # Add multiple weights
            weights = {}
            for i in range(20):
                data = np.random.randn(15, 15).astype(np.float32)
                weight = WeightTensor(
                    data=data,
                    metadata=WeightMetadata(
                        name=f"w_{i}",
                        shape=data.shape,
                        dtype=data.dtype
                    )
                )
                weights[f"w_{i}"] = weight
            
            repo.stage_weights(weights)
            repo.commit("Weights")
            
            # Run migration with custom options
            result = subprocess.run(
                [
                    "coral-ml", "cluster", "migrate",
                    "--strategy", "hierarchical",
                    "--threshold", "0.9",
                    "--batch-size", "5"
                ],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Strategy: hierarchical" in result.stdout
            assert "Threshold: 0.9" in result.stdout
            assert "Batch size: 5" in result.stdout
            assert "Migration complete:" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])