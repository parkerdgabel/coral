"""Test clustering CLI commands."""

import json
import pytest
from pathlib import Path
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from coral.cli.main import CoralCLI
from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.version_control.repository import Repository


class TestClusteringCLI:
    """Test clustering CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.cli = CoralCLI()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)
        
        # Initialize repository
        self.repo = Repository(self.repo_path, init=True)
        
        # Add some test weights
        weights = {}
        for i in range(10):
            data = np.random.randn(10, 10) + i * 0.1  # Similar weights
            weight = WeightTensor(
                data=data,
                metadata=WeightMetadata(
                    name=f"weight_{i}",
                    shape=data.shape,
                    dtype=data.dtype
                )
            )
            weights[f"weight_{i}"] = weight
        
        self.repo.stage_weights(weights)
        self.repo.commit("Initial weights")

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_cluster_analyze(self, capsys):
        """Test cluster analyze command."""
        # Test default format
        result = self.cli.run([
            "cluster", "analyze",
            "--threshold", "0.95"
        ])
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Analyzing repository for clustering opportunities" in captured.out
        assert "Total weights:" in captured.out
        assert "Potential clusters:" in captured.out
        
        # Test JSON format
        result = self.cli.run([
            "cluster", "analyze",
            "--format", "json"
        ])
        assert result == 0
        
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_weights" in data
        assert "potential_clusters" in data
        
        # Test CSV format
        result = self.cli.run([
            "cluster", "analyze",
            "--format", "csv"
        ])
        assert result == 0
        
        captured = capsys.readouterr()
        assert "metric,value" in captured.out
        assert "total_weights," in captured.out

    def test_cluster_status(self, capsys):
        """Test cluster status command."""
        # Status before clustering
        result = self.cli.run(["cluster", "status"])
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Clustering is not enabled" in captured.out
        
        # Mock clustering enabled
        with patch.object(Repository, 'get_clustering_status') as mock_status:
            mock_status.return_value = {
                'enabled': True,
                'strategy': 'kmeans',
                'num_clusters': 5,
                'clustered_weights': 10,
                'space_saved_bytes': 1024 * 1024,
                'reduction_percentage': 0.25,
                'last_updated': '2023-01-01 00:00:00',
                'cluster_health': {
                    'healthy': 4,
                    'warnings': 1,
                    'errors': 0
                }
            }
            
            result = self.cli.run(["cluster", "status"])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Clustering Status:" in captured.out
            assert "Strategy:          kmeans" in captured.out
            assert "Clusters:          5" in captured.out
            assert "Space saved:       1.0 MB" in captured.out

    def test_cluster_create(self, capsys):
        """Test cluster create command."""
        # Test dry run
        with patch.object(Repository, 'analyze_clustering') as mock_analyze:
            mock_analyze.return_value = {
                'potential_clusters': 3,
                'estimated_reduction': 0.3
            }
            
            result = self.cli.run([
                "cluster", "create",
                "--dry-run"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "DRY RUN:" in captured.out
            assert "Estimated clusters: 3" in captured.out
            assert "Estimated reduction: 30.0%" in captured.out
        
        # Test actual creation
        with patch.object(Repository, 'create_clusters') as mock_create:
            mock_create.return_value = {
                'num_clusters': 3,
                'weights_clustered': 10,
                'space_saved': 512 * 1024,
                'reduction_percentage': 0.25,
                'time_elapsed': 1.5
            }
            
            result = self.cli.run([
                "cluster", "create", "kmeans",
                "--levels", "2",
                "--threshold", "0.95"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Creating clusters using kmeans strategy" in captured.out
            assert "Created clusters:    3" in captured.out
            assert "Space saved:         0.5 MB" in captured.out
            
            # Verify parameters passed
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs['strategy'] == 'kmeans'
            assert call_kwargs['levels'] == 2
            assert call_kwargs['similarity_threshold'] == 0.95

    def test_cluster_optimize(self, capsys):
        """Test cluster optimize command."""
        with patch.object(Repository, 'optimize_clusters') as mock_optimize:
            mock_optimize.return_value = {
                'clusters_optimized': 3,
                'weights_moved': 5,
                'space_saved': 256 * 1024,
                'quality_improvement': 0.05,
                'warnings': ['Some weights could not be optimized']
            }
            
            result = self.cli.run([
                "cluster", "optimize",
                "--aggressive",
                "--target-reduction", "0.4"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Optimizing clusters" in captured.out
            assert "Clusters optimized:  3" in captured.out
            assert "Weights moved:       5" in captured.out
            assert "Space saved:         0.2 MB" in captured.out
            assert "Some weights could not be optimized" in captured.out

    def test_cluster_list(self, capsys):
        """Test cluster list command."""
        mock_clusters = [
            {
                'id': 'cluster_001',
                'size': 5,
                'quality': 0.95,
                'compression_ratio': 2.5,
                'centroid_hash': 'abcd1234efgh5678'
            },
            {
                'id': 'cluster_002',
                'size': 3,
                'quality': 0.88,
                'compression_ratio': 1.8,
                'centroid_hash': 'ijkl9012mnop3456'
            }
        ]
        
        with patch.object(Repository, 'list_clusters') as mock_list:
            mock_list.return_value = mock_clusters
            
            # Test text format
            result = self.cli.run(["cluster", "list"])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "cluster_001" in captured.out
            assert "cluster_002" in captured.out
            assert "Total clusters: 2" in captured.out
            
            # Test JSON format
            result = self.cli.run([
                "cluster", "list",
                "--format", "json"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) == 2
            assert data[0]['id'] == 'cluster_001'
            
            # Test CSV format
            result = self.cli.run([
                "cluster", "list",
                "--format", "csv",
                "--sort-by", "quality",
                "--limit", "1"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "cluster_id,size,quality,compression_ratio,centroid_hash" in captured.out

    def test_cluster_info(self, capsys):
        """Test cluster info command."""
        mock_info = {
            'id': 'cluster_001',
            'size': 5,
            'quality': 0.95,
            'compression_ratio': 2.5,
            'space_saved': 1024 * 512,
            'centroid_hash': 'abcd1234',
            'created': '2023-01-01 00:00:00',
            'statistics': {
                'mean_similarity': 0.96,
                'min_similarity': 0.94,
                'max_similarity': 0.98,
                'std_deviation': 0.015
            },
            'weights': [
                {'name': 'weight_1', 'hash': 'hash1234', 'similarity': 0.97},
                {'name': 'weight_2', 'hash': 'hash5678', 'similarity': 0.95}
            ]
        }
        
        with patch.object(Repository, 'get_cluster_info') as mock_get:
            mock_get.return_value = mock_info
            
            result = self.cli.run([
                "cluster", "info", "cluster_001",
                "--show-stats",
                "--show-weights"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Cluster: cluster_001" in captured.out
            assert "Size: 5 weights" in captured.out
            assert "Mean similarity: 0.960" in captured.out
            assert "weight_1 (hash1234) similarity: 0.970" in captured.out

    def test_cluster_export_import(self, capsys):
        """Test cluster export and import commands."""
        export_file = self.repo_path / "clusters.json"
        
        mock_config = {
            'version': '1.0',
            'strategy': 'kmeans',
            'clusters': [
                {'id': 'cluster_001', 'weights': ['w1', 'w2']}
            ]
        }
        
        # Test export
        with patch.object(Repository, 'export_clustering_config') as mock_export:
            mock_export.return_value = mock_config
            
            result = self.cli.run([
                "cluster", "export", str(export_file),
                "--format", "json",
                "--include-weights"
            ])
            assert result == 0
            assert export_file.exists()
            
            captured = capsys.readouterr()
            assert "Exporting clustering configuration" in captured.out
            assert "Configuration exported" in captured.out
        
        # Test import
        with patch.object(Repository, 'validate_clustering_config') as mock_validate, \
             patch.object(Repository, 'import_clustering_config') as mock_import:
            
            mock_validate.return_value = {'valid': True, 'errors': []}
            mock_import.return_value = {
                'clusters_imported': 1,
                'weights_assigned': 2,
                'conflicts_resolved': 0
            }
            
            result = self.cli.run([
                "cluster", "import", str(export_file),
                "--validate",
                "--merge"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Validating configuration" in captured.out
            assert "Configuration is valid" in captured.out
            assert "Clusters imported:   1" in captured.out

    def test_cluster_compare(self, capsys):
        """Test cluster compare command."""
        mock_comparison = {
            'commit1': {
                'hash': 'abcd1234efgh5678',
                'date': '2023-01-01',
                'num_clusters': 3,
                'clustered_weights': 10
            },
            'commit2': {
                'hash': 'ijkl9012mnop3456',
                'date': '2023-01-02',
                'num_clusters': 4,
                'clustered_weights': 12
            },
            'changes': {
                'clusters_added': 2,
                'clusters_removed': 1,
                'clusters_modified': 2
            },
            'weight_migrations': [
                {'weight': 'weight_1', 'from': 'cluster_001', 'to': 'cluster_002'}
            ]
        }
        
        with patch.object(Repository, 'compare_clustering') as mock_compare:
            mock_compare.return_value = mock_comparison
            
            result = self.cli.run([
                "cluster", "compare", "HEAD~1", "HEAD",
                "--show-migrations"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Comparing clustering: HEAD~1 vs HEAD" in captured.out
            assert "Clusters added:    2" in captured.out
            assert "weight_1: cluster_001 -> cluster_002" in captured.out

    def test_cluster_validate(self, capsys):
        """Test cluster validate command."""
        mock_validation = {
            'valid': False,
            'errors': ['Missing centroid for cluster_001'],
            'warnings': ['Low quality for cluster_002']
        }
        
        mock_fix = {
            'errors_fixed': 1,
            'errors_remaining': 0
        }
        
        with patch.object(Repository, 'validate_clusters') as mock_validate, \
             patch.object(Repository, 'fix_clustering_errors') as mock_fix_errors:
            
            mock_validate.return_value = mock_validation
            mock_fix_errors.return_value = mock_fix
            
            result = self.cli.run([
                "cluster", "validate",
                "--fix",
                "--strict"
            ])
            assert result == 0  # Because errors were fixed
            
            captured = capsys.readouterr()
            assert "Validating cluster integrity" in captured.out
            assert "Valid:     False" in captured.out
            assert "Missing centroid for cluster_001" in captured.out
            assert "Low quality for cluster_002" in captured.out
            assert "Attempting to fix errors" in captured.out
            assert "Fixed: 1 error(s)" in captured.out

    def test_cluster_benchmark(self, capsys):
        """Test cluster benchmark command."""
        with patch.object(Repository, 'create_clusters') as mock_create:
            # Mock results for different strategies
            mock_create.side_effect = [
                # KMeans results
                {'num_clusters': 3, 'quality': 0.95, 'reduction_percentage': 0.3},
                {'num_clusters': 3, 'quality': 0.94, 'reduction_percentage': 0.29},
                # Hierarchical results
                {'num_clusters': 4, 'quality': 0.92, 'reduction_percentage': 0.28},
                {'num_clusters': 4, 'quality': 0.93, 'reduction_percentage': 0.27}
            ]
            
            result = self.cli.run([
                "cluster", "benchmark",
                "--strategies", "kmeans", "hierarchical",
                "--iterations", "2"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Running clustering benchmarks" in captured.out
            assert "Benchmarking kmeans" in captured.out
            assert "Benchmarking hierarchical" in captured.out
            assert "Benchmark Summary:" in captured.out
            assert "kmeans" in captured.out
            assert "hierarchical" in captured.out

    def test_cluster_report(self, capsys):
        """Test cluster report command."""
        mock_report = {
            'timestamp': '2023-01-01 00:00:00',
            'repository': str(self.repo_path),
            'overview': {
                'total_clusters': 3,
                'clustered_weights': 10,
                'space_saved': 1024 * 1024,
                'reduction_percentage': 0.25,
                'average_quality': 0.93
            },
            'top_clusters': [
                {
                    'id': 'cluster_001',
                    'size': 5,
                    'quality': 0.95,
                    'space_saved': 512 * 1024
                }
            ],
            'clusters': [
                {
                    'id': 'cluster_001',
                    'size': 5,
                    'quality': 0.95,
                    'compression_ratio': 2.5,
                    'space_saved': 512 * 1024
                }
            ]
        }
        
        with patch.object(Repository, 'generate_clustering_report') as mock_report_gen:
            mock_report_gen.return_value = mock_report
            
            # Test text format
            result = self.cli.run([
                "cluster", "report",
                "--verbose"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Coral Clustering Report" in captured.out
            assert "Total clusters:      3" in captured.out
            assert "Space saved:         1.0 MB" in captured.out
            
            # Test HTML format with output file
            output_file = self.repo_path / "report.html"
            result = self.cli.run([
                "cluster", "report",
                "--format", "html",
                "--output", str(output_file)
            ])
            assert result == 0
            assert output_file.exists()
            
            content = output_file.read_text()
            assert "<html>" in content
            assert "Coral Clustering Report" in content

    def test_cluster_rebalance(self, capsys):
        """Test cluster rebalance command."""
        with patch.object(Repository, 'rebalance_clusters') as mock_rebalance:
            mock_rebalance.return_value = {
                'clusters_merged': 2,
                'clusters_split': 1,
                'weights_reassigned': 7,
                'balance_score': 0.85
            }
            
            result = self.cli.run([
                "cluster", "rebalance",
                "--max-cluster-size", "10",
                "--min-cluster-size", "3"
            ])
            assert result == 0
            
            captured = capsys.readouterr()
            assert "Rebalancing clusters" in captured.out
            assert "Clusters merged:     2" in captured.out
            assert "Clusters split:      1" in captured.out
            assert "Weights reassigned:  7" in captured.out
            assert "Balance score:       0.850" in captured.out

    def test_cluster_help(self, capsys):
        """Test cluster help command."""
        # Help command causes SystemExit(0), so we need to catch it
        with pytest.raises(SystemExit) as exc_info:
            self.cli.run(["cluster", "--help"])
        
        assert exc_info.value.code == 0
        
        captured = capsys.readouterr()
        assert "Clustering commands" in captured.out
        assert "analyze" in captured.out
        assert "create" in captured.out
        assert "optimize" in captured.out
        assert "list" in captured.out

    def test_error_handling(self, capsys):
        """Test error handling in cluster commands."""
        # Test cluster not found
        with patch.object(Repository, 'get_cluster_info') as mock_get:
            mock_get.return_value = None
            
            result = self.cli.run(["cluster", "info", "nonexistent"])
            assert result == 1
            
            captured = capsys.readouterr()
            assert "Error: Cluster 'nonexistent' not found" in captured.err
        
        # Test import file not found
        result = self.cli.run([
            "cluster", "import", "/nonexistent/file.json"
        ])
        assert result == 1
        
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.err
        
        # Test validation failure
        with patch.object(Repository, 'validate_clustering_config') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Invalid cluster format']
            }
            
            # Create a dummy config file
            config_file = self.repo_path / "bad_config.json"
            config_file.write_text('{"clusters": []}')
            
            result = self.cli.run([
                "cluster", "import", str(config_file),
                "--validate"
            ])
            assert result == 1
            
            captured = capsys.readouterr()
            assert "Configuration validation failed" in captured.out
            assert "Invalid cluster format" in captured.out


@pytest.mark.parametrize("command,expected_methods", [
    (["cluster", "analyze"], ["analyze_clustering"]),
    (["cluster", "status"], ["get_clustering_status"]),
    (["cluster", "create"], ["create_clusters"]),
    (["cluster", "optimize"], ["optimize_clusters"]),
    (["cluster", "list"], ["list_clusters"]),
    (["cluster", "validate"], ["validate_clusters"])
])
def test_command_method_mapping(command, expected_methods):
    """Test that CLI commands call the correct repository methods."""
    cli = CoralCLI()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        Repository(repo_path, init=True)
        
        # Mock all expected methods
        with patch.multiple(Repository, **{
            method: MagicMock(return_value={
                'enabled': True,
                'valid': True,
                'num_clusters': 0,
                'clusters': []
            }) for method in expected_methods
        }):
            result = cli.run(command)
            
            # Verify the expected methods were called
            for method in expected_methods:
                getattr(Repository, method).assert_called()