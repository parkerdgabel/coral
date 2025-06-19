"""
Example demonstrating PyTorch training with Coral version control.
"""

from pathlib import Path

# Coral imports
from coral import Repository
from coral.integrations.pytorch import CoralTrainer
from coral.training import CheckpointConfig

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy dataset for training."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def train_with_coral():
    """Example training loop with Coral integration."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        return

    # Setup repository
    repo_path = Path("./coral_pytorch_demo")
    repo_path.mkdir(exist_ok=True)

    # Initialize or load repository
    try:
        repo = Repository(repo_path)
        print("Loaded existing repository")
    except ValueError:
        repo = Repository(repo_path, init=True)
        print("Initialized new repository")

    # Create model
    model = SimpleModel()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Setup training components
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Create dataset
    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Configure checkpointing
    checkpoint_config = CheckpointConfig(
        save_every_n_epochs=2,
        save_on_best_metric="accuracy",
        minimize_metric=False,  # Higher accuracy is better
        keep_last_n_checkpoints=5,
        keep_best_n_checkpoints=3,
        auto_commit=True,
        tag_best_checkpoints=True,
    )

    # Initialize Coral trainer
    trainer = CoralTrainer(
        model=model,
        repository=repo,
        experiment_name="simple_model_training",
        checkpoint_config=checkpoint_config,
    )

    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)

    # Add callbacks
    def on_epoch_end(trainer):
        print(f"Epoch {trainer.current_epoch} completed")
        print(f"  Loss: {trainer.training_metrics.get('loss', 0):.4f}")
        print(f"  Accuracy: {trainer.training_metrics.get('accuracy', 0):.4f}")
        print(f"  Learning rate: {trainer._get_learning_rate():.6f}")

    def on_checkpoint_save(trainer, commit_hash):
        print(f"  💾 Checkpoint saved: {commit_hash[:8]}")

    trainer.add_callback("epoch_end", on_epoch_end)
    trainer.add_callback("checkpoint_save", on_checkpoint_save)

    # Training loop
    print("\\nStarting training...")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Update trainer with step info
            if batch_idx % 10 == 0:
                trainer.step(loss=loss.item(), accuracy=correct / total)

        # End of epoch
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total

        trainer.update_metrics(loss=epoch_loss, accuracy=epoch_accuracy)

        trainer.epoch_end(epoch)

    print("\\nTraining completed!")

    # Show training summary
    summary = trainer.get_training_summary()
    print("\\nTraining Summary:")
    for key, value in summary.items():
        if key != "metrics":
            print(f"  {key}: {value}")

    print("\\nFinal Metrics:")
    for key, value in summary["metrics"].items():
        print(f"  {key}: {value:.4f}")

    # List checkpoints
    checkpoints = trainer.list_checkpoints()
    print(f"\\nSaved {len(checkpoints)} checkpoints:")
    for checkpoint in checkpoints[-5:]:  # Show last 5
        print(f"  Epoch {checkpoint['epoch']}: {checkpoint['commit_hash'][:8]}")
        if checkpoint.get("is_best"):
            print("    ^ Best checkpoint")

    # Demonstrate checkpoint loading
    print("\\nTesting checkpoint loading...")

    # Save current state
    current_params = {name: param.clone() for name, param in model.named_parameters()}

    # Load best checkpoint
    success = trainer.load_checkpoint(load_best=True)
    if success:
        print("✅ Successfully loaded best checkpoint")

        # Verify parameters changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, current_params[name]):
                changed = True
                break

        if changed:
            print("✅ Model parameters were restored")
        else:
            print("⚠️  Model parameters unchanged (might be the same checkpoint)")
    else:
        print("❌ Failed to load checkpoint")

    print("\\nDemonstration complete!")
    print(f"Repository created at: {repo_path}")
    print("\\nYou can explore the repository with CLI commands:")
    print(f"  cd {repo_path}")
    print("  coral log")
    print("  coral branch")
    print("  coral status")


def demonstrate_cli_usage():
    """Show how to use Coral CLI with PyTorch models."""
    print("\\n" + "=" * 50)
    print("CLI USAGE EXAMPLE")
    print("=" * 50)

    print("""
After training, you can use Coral CLI commands:

1. Initialize repository:
   coral init my_project

2. Save model weights (after converting to .npy files):
   coral add model_weights.npz
   coral commit -m "Initial model checkpoint"

3. View history:
   coral log --oneline

4. Create branches for experiments:
   coral branch experiment_lr_0.01
   coral checkout experiment_lr_0.01

5. Merge successful experiments:
   coral checkout main
   coral merge experiment_lr_0.01

6. Tag successful models:
   coral tag v1.0 -d "Production model" --metric accuracy=0.95

7. Compare different versions:
   coral diff v0.9 v1.0

8. Clean up old checkpoints:
   coral gc

For more commands: coral --help
""")


if __name__ == "__main__":
    train_with_coral()
    demonstrate_cli_usage()
