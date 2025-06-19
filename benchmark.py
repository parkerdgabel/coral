# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "h5py",
#   "xxhash",
#   "tqdm",
#   "torch",
# ]
# ///

import os
import sys
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

# Add src to path to import coral
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from coral.core.weight_tensor import WeightTensor, WeightMetadata
from coral.core.deduplicator import Deduplicator
from coral.storage.hdf5_store import HDF5Store
from coral.version_control.repository import Repository


def create_test_models() -> List[nn.Module]:
    """Create a variety of test models with different characteristics."""
    models = []
    
    # Small CNN
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Medium MLP
    class MediumMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create base models
    models.append(("SmallCNN", SmallCNN()))
    models.append(("MediumMLP", MediumMLP()))
    
    # Create variations with small perturbations (similar weights)
    for name, model in models[:]:
        for i in range(5):  # More variations to show deduplication better
            variation = type(model)()
            # Add very small noise to create highly similar weights
            with torch.no_grad():
                for param, orig_param in zip(variation.parameters(), model.parameters()):
                    if i == 0:
                        # Nearly identical (99.9% similar)
                        noise = torch.randn_like(param) * 0.001
                    elif i == 1:
                        # Very similar (99% similar)
                        noise = torch.randn_like(param) * 0.01
                    else:
                        # Moderately similar (95% similar)
                        noise = torch.randn_like(param) * 0.05
                    param.copy_(orig_param + noise)
            models.append((f"{name}_var{i+1}", variation))
        
        # Create some checkpoints (identical weights)
        for i in range(3):
            checkpoint = type(model)()
            with torch.no_grad():
                for param, orig_param in zip(checkpoint.parameters(), model.parameters()):
                    param.copy_(orig_param)  # Exact copy
            models.append((f"{name}_checkpoint{i+1}", checkpoint))
    
    return models


def measure_naive_storage(models: List[Tuple[str, nn.Module]]) -> Tuple[int, Dict[str, int]]:
    """Measure storage size without any deduplication."""
    total_size = 0
    model_sizes = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, model in models:
            model_path = Path(temp_dir) / f"{name}.pth"
            torch.save(model.state_dict(), model_path)
            size = model_path.stat().st_size
            total_size += size
            model_sizes[name] = size
    
    return total_size, model_sizes


def measure_coral_storage(models: List[Tuple[str, nn.Module]]) -> Tuple[int, Dict[str, Dict], Dict]:
    """Measure storage size with Coral's deduplication and delta encoding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "coral_repo"
        repo = Repository(repo_path, init=True)
        
        model_stats = {}
        
        for name, model in models:
            weights_dict = {}
            param_count = 0
            
            for param_name, param in model.named_parameters():
                param_data = param.detach().numpy()
                metadata = WeightMetadata(
                    name=f"{name}.{param_name}",
                    shape=param_data.shape,
                    dtype=param_data.dtype,
                    model_name=name,
                    layer_type=param_name.split('.')[-1]  # weight, bias, etc.
                )
                weight_tensor = WeightTensor(
                    data=param_data,
                    metadata=metadata
                )
                weights_dict[f"{name}.{param_name}"] = weight_tensor
                param_count += param.numel()
            
            # Stage and commit weights in repository
            repo.stage_weights(weights_dict)
            repo.commit(f"Add {name}")
            
            model_stats[name] = {
                "param_count": param_count,
                "weight_count": len(weights_dict)
            }
        
        # Get storage statistics
        store_path = repo_path / ".coral" / "objects" / "weights.h5"
        storage_size = store_path.stat().st_size if store_path.exists() else 0
        
        # Get deduplication statistics (if available)
        try:
            dedup_stats = repo.store.get_storage_stats() if hasattr(repo, 'store') else {}
        except:
            dedup_stats = {}
        
        # Calculate our own basic statistics
        total_weight_count = sum(stats["weight_count"] for stats in model_stats.values())
        total_param_count = sum(stats["param_count"] for stats in model_stats.values())
        
        dedup_stats.update({
            "total_models": len(models),
            "total_weight_tensors": total_weight_count,
            "total_parameters": total_param_count
        })
        
        return storage_size, model_stats, dedup_stats


def run_benchmark() -> None:
    """Run comprehensive storage benchmarking."""
    print(f"{Colors.BOLD}{Colors.HEADER}🚀 Coral Storage Benchmarking{Colors.ENDC}")
    print(f"{Colors.BLUE}{'=' * 50}{Colors.ENDC}")
    
    # Create test models
    print(f"\n{Colors.CYAN}📦 Creating test models...{Colors.ENDC}")
    models = create_test_models()
    print(f"   Created {Colors.GREEN}{len(models)}{Colors.ENDC} models")
    
    # Measure naive storage
    print(f"\n{Colors.CYAN}📏 Measuring naive storage (PyTorch .pth files)...{Colors.ENDC}")
    start_time = time.time()
    naive_total, naive_sizes = measure_naive_storage(models)
    naive_time = time.time() - start_time
    
    print(f"   Total size: {Colors.YELLOW}{naive_total:,}{Colors.ENDC} bytes ({Colors.YELLOW}{naive_total / 1024 / 1024:.2f} MB{Colors.ENDC})")
    print(f"   Time: {Colors.YELLOW}{naive_time:.2f}{Colors.ENDC} seconds")
    
    # Measure Coral storage
    print(f"\n{Colors.CYAN}🪸 Measuring Coral storage (with deduplication + delta encoding)...{Colors.ENDC}")
    start_time = time.time()
    coral_total, model_stats, dedup_stats = measure_coral_storage(models)
    coral_time = time.time() - start_time
    
    print(f"   Total size: {Colors.GREEN}{coral_total:,}{Colors.ENDC} bytes ({Colors.GREEN}{coral_total / 1024 / 1024:.2f} MB{Colors.ENDC})")
    print(f"   Time: {Colors.GREEN}{coral_time:.2f}{Colors.ENDC} seconds")
    
    # Calculate savings
    space_saved = naive_total - coral_total
    compression_ratio = naive_total / coral_total if coral_total > 0 else float('inf')
    space_savings_percent = (space_saved / naive_total * 100) if naive_total > 0 else 0
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}📊 Results Summary{Colors.ENDC}")
    print(f"{Colors.BLUE}{'=' * 50}{Colors.ENDC}")
    print(f"Naive storage:     {Colors.YELLOW}{naive_total:,}{Colors.ENDC} bytes ({Colors.YELLOW}{naive_total / 1024 / 1024:.2f} MB{Colors.ENDC})")
    print(f"Coral storage:     {Colors.GREEN}{coral_total:,}{Colors.ENDC} bytes ({Colors.GREEN}{coral_total / 1024 / 1024:.2f} MB{Colors.ENDC})")
    print(f"Space saved:       {Colors.BOLD}{Colors.GREEN}{space_saved:,}{Colors.ENDC} bytes ({Colors.BOLD}{Colors.GREEN}{space_saved / 1024 / 1024:.2f} MB{Colors.ENDC})")
    print(f"Compression ratio: {Colors.BOLD}{Colors.GREEN}{compression_ratio:.2f}x{Colors.ENDC}")
    print(f"Space savings:     {Colors.BOLD}{Colors.GREEN}{space_savings_percent:.1f}%{Colors.ENDC}")
    
    print(f"\n{Colors.CYAN}Time comparison:{Colors.ENDC}")
    print(f"Naive:  {Colors.YELLOW}{naive_time:.2f}{Colors.ENDC} seconds")
    print(f"Coral:  {Colors.GREEN}{coral_time:.2f}{Colors.ENDC} seconds")
    overhead_percent = ((coral_time - naive_time) / naive_time * 100)
    overhead_color = Colors.GREEN if overhead_percent < 50 else Colors.YELLOW if overhead_percent < 100 else Colors.RED
    print(f"Overhead: {overhead_color}{overhead_percent:.1f}%{Colors.ENDC}")
    
    # Deduplication details
    if dedup_stats:
        print(f"\n{Colors.BOLD}{Colors.CYAN}🔍 Deduplication Details{Colors.ENDC}")
        print(f"{Colors.BLUE}{'=' * 30}{Colors.ENDC}")
        print(f"Total models:            {Colors.GREEN}{dedup_stats.get('total_models', 'N/A')}{Colors.ENDC}")
        print(f"Total weight tensors:    {Colors.GREEN}{dedup_stats.get('total_weight_tensors', 'N/A')}{Colors.ENDC}")
        print(f"Total parameters:        {Colors.GREEN}{dedup_stats.get('total_parameters', 'N/A'):,}{Colors.ENDC}")
        print(f"Unique weights stored:   {Colors.GREEN}{dedup_stats.get('unique_weights', 'N/A')}{Colors.ENDC}")
        print(f"Delta-encoded weights:   {Colors.GREEN}{dedup_stats.get('delta_weights', 'N/A')}{Colors.ENDC}")
        if 'dedup_ratio' in dedup_stats:
            print(f"Deduplication ratio:     {Colors.BOLD}{Colors.GREEN}{dedup_stats.get('dedup_ratio'):.2f}x{Colors.ENDC}")
    
    # Model breakdown
    print(f"\n{Colors.BOLD}{Colors.CYAN}📋 Per-Model Breakdown{Colors.ENDC}")
    print(f"{Colors.BLUE}{'=' * 40}{Colors.ENDC}")
    for name, size in naive_sizes.items():
        stats = model_stats.get(name, {})
        name_color = Colors.GREEN if 'checkpoint' in name else Colors.YELLOW if 'var' in name else Colors.CYAN
        print(f"{name_color}{name:15}{Colors.ENDC} | {size:8,} bytes | {stats.get('param_count', 0):8,} params")


def main() -> None:
    """Main benchmarking entry point."""
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Benchmark interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Benchmark failed: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
