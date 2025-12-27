#!/usr/bin/env python3
"""
Demonstration of the Checkpoint Callback System.

This example shows how to use the checkpoint callback system to:
1. Register custom callbacks that execute after checkpoint saves
2. Handle callback errors gracefully
3. Manage callbacks (register, unregister, clear)
4. Use callbacks for custom logging, notifications, or other actions
"""

import tempfile
from pathlib import Path

import numpy as np

from coral.core.weight_tensor import WeightMetadata, WeightTensor
from coral.training.checkpoint_manager import CheckpointConfig, CheckpointManager
from coral.training.training_state import TrainingState
from coral.version_control.repository import Repository


def custom_logger_callback(state: TrainingState, commit_hash: str):
    """Custom callback that logs checkpoint information."""
    print(f"📝 Custom Logger: Checkpoint saved at epoch {state.epoch}")
    print(f"   Step: {state.global_step}, Loss: {state.loss:.4f}")
    if commit_hash:
        print(f"   Commit hash: {commit_hash[:8]}...")
    print()


def metrics_tracker_callback(state: TrainingState, commit_hash: str):
    """Callback that tracks metrics in a custom format."""
    accuracy = state.metrics.get("accuracy", 0.0)
    val_loss = state.metrics.get("val_loss", 0.0)
    print("📊 Metrics Tracker:")
    print(f"   Training Loss: {state.loss:.4f}")
    print(f"   Accuracy: {accuracy:.3%}")
    print(f"   Validation Loss: {val_loss:.4f}")
    print()


def notification_callback(state: TrainingState, commit_hash: str):
    """Callback that could send notifications (simulated here)."""
    if state.metrics.get("accuracy", 0) > 0.95:
        print("🎉 Achievement Unlocked: >95% accuracy reached!")
        print("   (In real use, this could send an email or Slack message)")
    print()


def problematic_callback(state: TrainingState, commit_hash: str):
    """A callback that raises an error to demonstrate error handling."""
    if state.epoch == 2:
        raise RuntimeError("Simulated callback error on epoch 2")
    print("✅ Problematic callback executed successfully")


def main():
    """Demonstrate the checkpoint callback system."""
    print("=== Coral Checkpoint Callback System Demo ===\n")

    # Create a temporary repository
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        repo = Repository(repo_path, init=True)

        # Configure checkpoint manager
        config = CheckpointConfig(
            save_every_n_epochs=1,
            auto_commit=True,
            save_on_best_metric="accuracy",
            minimize_metric=False,  # We want to maximize accuracy
        )

        manager = CheckpointManager(
            repository=repo,
            config=config,
            model_name="DemoModel",
            experiment_name="callback_demo",
        )

        # Create sample weights
        weights = {
            "layer1.weight": WeightTensor(
                data=np.random.randn(5, 3).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.weight",
                    shape=(5, 3),
                    dtype=np.float32,
                    layer_type="Linear",
                ),
            ),
            "layer1.bias": WeightTensor(
                data=np.random.randn(5).astype(np.float32),
                metadata=WeightMetadata(
                    name="layer1.bias",
                    shape=(5,),
                    dtype=np.float32,
                    layer_type="Linear",
                ),
            ),
        }

        print("1. Registering callbacks...")
        manager.register_checkpoint_callback(custom_logger_callback)
        manager.register_checkpoint_callback(metrics_tracker_callback)
        manager.register_checkpoint_callback(notification_callback)
        manager.register_checkpoint_callback(problematic_callback)

        print(f"   Registered callbacks: {manager.list_callbacks()}")
        print()

        print("2. Training simulation with callbacks...")
        for epoch in range(1, 4):
            # Simulate training progress
            loss = max(0.1, 1.0 - epoch * 0.3)  # Decreasing loss
            accuracy = min(0.99, 0.7 + epoch * 0.1)  # Increasing accuracy
            val_loss = loss + 0.05  # Slightly higher validation loss

            state = TrainingState(
                epoch=epoch,
                global_step=epoch * 100,
                learning_rate=0.01,
                loss=loss,
                metrics={
                    "accuracy": accuracy,
                    "val_loss": val_loss,
                },
            )

            print(f"--- Epoch {epoch} ---")
            commit_hash = manager.save_checkpoint(weights, state, force=True)
            commit_short = commit_hash[:8] if commit_hash else "None"
            print(f"Checkpoint saved with commit: {commit_short}")
            print()

        print("3. Managing callbacks...")
        print(f"   Current callbacks: {manager.list_callbacks()}")

        # Remove the problematic callback
        removed = manager.unregister_checkpoint_callback(problematic_callback)
        print(f"   Removed problematic callback: {removed}")
        print(f"   Remaining callbacks: {manager.list_callbacks()}")

        # Save one more checkpoint without the problematic callback
        print("\n4. Final checkpoint without problematic callback...")
        final_state = TrainingState(
            epoch=4,
            global_step=400,
            learning_rate=0.01,
            loss=0.05,
            metrics={"accuracy": 0.98, "val_loss": 0.08},
        )
        manager.save_checkpoint(weights, final_state, force=True)

        # Clear all callbacks
        cleared_count = manager.clear_callbacks()
        print(f"\n5. Cleared {cleared_count} callbacks")
        print(f"   Remaining callbacks: {manager.list_callbacks()}")

        # Save final checkpoint with no callbacks
        print("\n6. Final checkpoint with no callbacks (should be silent)...")
        final_state.epoch = 5
        manager.save_checkpoint(weights, final_state, force=True)
        print("   Checkpoint saved with no callback output")

        print("\n=== Demo Complete ===")
        print("The callback system allows you to:")
        print("• Execute custom code after each checkpoint save")
        print("• Handle errors gracefully without breaking training")
        print("• Register multiple callbacks for different purposes")
        print("• Manage callbacks dynamically during training")


if __name__ == "__main__":
    main()
