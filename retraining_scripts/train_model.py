"""
Model training script for bomb detection
Based on CNN2_autokeras notebook
"""

import pickle
import numpy as np
import random
import datetime
import os
from pathlib import Path
from typing import Optional

# GPU configuration for Lightning Studio
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is visible

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable

# Improved GPU configuration
print("TensorFlow version:", tf.__version__)
print("CUDA support built:", tf.test.is_built_with_cuda())
print("GPU support built:", tf.test.is_built_with_gpu_support())

# Check physical devices
physical_devices = tf.config.list_physical_devices()
print("All physical devices:", physical_devices)

gpus = tf.config.list_physical_devices("GPU")
print(f"Found GPUs: {gpus}")

if gpus:
    try:
        # Enable memory growth and set as logical device
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Set the first GPU as the default device
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"Logical GPUs: {logical_gpus}")

        # Test GPU with a simple operation
        with tf.device("/GPU:0"):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print("GPU test successful:", result.numpy())

        print(f"✅ GPU configuration successful - using {len(gpus)} GPU(s)")
        GPU_AVAILABLE = True

    except RuntimeError as e:
        print(f"❌ GPU configuration failed: {e}")
        print("Falling back to CPU")
        GPU_AVAILABLE = False
else:
    print("❌ No GPUs detected by TensorFlow")
    # Additional diagnostics
    print("Checking NVIDIA-SMI...")
    os.system("nvidia-smi")
    GPU_AVAILABLE = False

# AutoKeras imports
import autokeras
import keras_tuner

# Temporary workaround for AutoKeras serialization issues
from autokeras.preprocessors.postprocessors import SigmoidPostprocessor

register_keras_serializable(package="AutoKeras")(SigmoidPostprocessor)

# Scikit learn functions
from sklearn.model_selection import train_test_split

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set up matplotlib
mpl.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class SaturationLogger(keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        x = self.x_val
        if x.shape[0] > 10000:
            idx = np.random.choice(x.shape[0], 10000, replace=False)
            x = x[idx]
        preds = self.model.predict(x, verbose=0).ravel()
        frac_hi = np.mean(preds > 0.95)
        frac_lo = np.mean(preds < 0.05)
        mean_p = float(np.mean(preds))
        print(
            f"\n[Saturation] epoch {epoch+1}: "
            f"frac>0.95={frac_hi:.3f}, frac<0.05={frac_lo:.3f}, mean(p)={mean_p:.3f}"
        )
        if frac_hi > 0.98 or frac_lo > 0.98:
            print("[Saturation] Predictions saturated. Stopping training.")
            self.model.stop_training = True


class ModelTrainer:

    def __init__(
        self,
        data_dir: str = "../data",
        train_pickle: Optional[str] = None,
        test_pickle: Optional[str] = None,
        models_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
    ):
        """Initialize ModelTrainer with configurable I/O paths.

        Args:
            data_dir: Base data directory (default: "../data")
            train_pickle: Path to training features pickle file (default: train_features_labels_2.pickle)
            test_pickle: Path to test features pickle file (default: test_features_labels_2.pickle)
            models_dir: Directory to save trained models (default: ../models)
            logs_dir: Directory to save training logs (default: ../logs)
        """
        self.data_dir = Path(data_dir)

        # Set paths to pickle files
        self.train_pickle_file = train_pickle or "train_features_labels.pickle"
        self.test_pickle_file = test_pickle or "test_features_labels.pickle"

        # Training parameters (optimized for GPU)
        self.EPOCHS = 100
        self.BATCH_SIZE = 32  # Adjust batch size based on GPU
        self.MAX_TRIALS = 50
        self.MAX_MODEL_SIZE = 1000000

        # Set seed for reproducible results
        self.seed = 123
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)

        # Metrics (same as notebook)
        self.METRICS = [
            keras.metrics.TruePositives(name="tp"),
            keras.metrics.FalsePositives(name="fp"),
            keras.metrics.TrueNegatives(name="tn"),
            keras.metrics.FalseNegatives(name="fn"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="prc", curve="PR"),
        ]

        # Create output dirs
        self.model_dir = Path(models_dir) if models_dir else Path("../models")
        self.logs_dir = Path(logs_dir) if logs_dir else Path("../logs")
        self.model_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load training and test data from pickle files."""
        print("=== LOADING DATA ===")

        # Load training data
        train_pickle_path = self.data_dir / self.train_pickle_file
        with open(train_pickle_path, "rb") as f:
            train_features, train_labels, input_shape = pickle.load(f)

        print(f"Training data loaded: {len(train_labels)} samples")
        print(f"Input shape: {input_shape}")
        print(f"Bomb files: {len(np.where(train_labels == 1)[0])}")

        # Load test data
        test_pickle_path = self.data_dir / self.test_pickle_file
        with open(test_pickle_path, "rb") as f:
            test_features, test_labels, input_shape = pickle.load(f)

        print(f"Test data loaded: {len(test_labels)} samples")
        print(f"Bomb files: {len(np.where(test_labels == 1)[0])}")

        return train_features, train_labels, test_features, test_labels, input_shape

    def calculate_class_weights(self, train_labels):
        """Calculate class weights for imbalanced data (same as notebook)."""
        print("\n=== CALCULATING CLASS WEIGHTS ===")

        pos = len(np.where(train_labels == 1)[0])
        neg = len(np.where(train_labels == 0)[0])

        # count the ratio of bombs to no bombs
        total = neg + pos
        print(
            "Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
                total, pos, 100 * pos / total
            )
        )

        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        print("Weight for class 0: {:.2f}".format(weight_for_0))
        print("Weight for class 1: {:.2f}".format(weight_for_1))

        return class_weight

    def setup_callbacks(self):
        """Setup training callbacks (same as notebook)."""
        print("\n=== SETTING UP CALLBACKS ===")

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_prc",
            verbose=1,
            patience=10,
            mode="max",
            restore_best_weights=True,
        )

        # TensorBoard
        logdir = self.logs_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        return early_stopping, tensorboard_callback

    def create_autokeras_model(self, input_shape):
        """Create AutoKeras model search space (optimized for GPU)."""
        print("\n=== CREATING AUTOKERAS MODEL ===")

        # Define the model space
        image_input = autokeras.ImageInput()

        # Conv layers - autokeras will test num layers, kernel size and count
        conv_layers = autokeras.ConvBlock()(image_input)

        # Dense layers - autokeras will test out num and width of dense layers
        dense_layers = autokeras.DenseBlock(use_batchnorm=False)(conv_layers)

        # Output node - we set up loss, classes and other params
        classification_head = autokeras.ClassificationHead(
            num_classes=1,
            multi_label=False,
            loss="binary_crossentropy",
            metrics=self.METRICS,
            dropout=None,  # would like to implement bias initializer but haven't figured this out
        )(dense_layers)

        device_info = (
            f"GPU (batch_size={self.BATCH_SIZE})"
            if GPU_AVAILABLE
            else f"CPU (batch_size={self.BATCH_SIZE})"
        )
        print(f"Using {device_info} AutoKeras configuration")
        print(f"Settings: max_trials={self.MAX_TRIALS}")

        # Build the model
        clf = autokeras.AutoModel(
            inputs=image_input,
            outputs=classification_head,
            objective=keras_tuner.Objective("val_prc", direction="max"),
            tuner="greedy",
            max_model_size=self.MAX_MODEL_SIZE,
            max_trials=self.MAX_TRIALS,
            seed=self.seed,
            directory=str(self.data_dir),
            overwrite=True,
            project_name="autokeras_bomb_detector",
        )

        return clf

    def train_autokeras_model(
        self,
        clf,
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_weight,
        callbacks,
    ):
        """Train the AutoKeras model (optimized for GPU)."""
        print("\n=== TRAINING AUTOKERAS MODEL ===")
        print(f"Hardware: {'GPU' if GPU_AVAILABLE else 'CPU'}")

        # Search stage - no early stopping to let AutoKeras handle its own early stop
        search_history = clf.fit(
            train_features,
            train_labels,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=[],  # No callbacks during search - AutoKeras handles its own
            validation_data=(val_features, val_labels),
            class_weight=class_weight,
        )

        return clf, search_history

    def export_and_retrain_model(
        self,
        clf,
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_weight,
        callbacks,
    ):
        """Export best model and retrain for final version (same as notebook)."""
        print("\n=== EXPORTING AND RETRAINING BEST MODEL ===")

        # Export best model
        best_model = clf.export_model()
        print("Best model architecture:")
        print(best_model.summary())

        # FIXED: Freeze batch normalization for stability
        for layer in best_model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        print("Frozen batch normalization layers for stability")

        # Set checkpoint path
        checkpoint_path = self.model_dir / "best_model_checkpoint"
        print(f"Will save model checkpoint to: {checkpoint_path}")

        # Model checkpoint callback - FIXED: monitor AUC for better stability
        save_model = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_prc",
            save_best_only=True,
            verbose=1,
        )

        # Compile the final model
        optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        best_model.compile(
            optimizer,
            loss="binary_crossentropy",
            metrics=self.METRICS,
        )

        # Retrain the autokeras best model
        final_model_history = best_model.fit(
            train_features,
            train_labels,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=[
                callbacks[0],
                save_model,
                callbacks[1],
            ],  # early_stopping, save_model, tensorboard
            validation_data=(val_features, val_labels),
            class_weight=class_weight,
        )

        return best_model, final_model_history

    def plot_training_history(self, history):
        """Plot training history (same as notebook)."""
        print("\n=== PLOTTING TRAINING HISTORY ===")

        def plot_metrics(history):
            metrics = ["loss", "prc", "precision", "recall"]
            for n, metric in enumerate(metrics):
                name = metric.replace("_", " ").capitalize()
                plt.subplot(2, 2, n + 1)
                plt.plot(
                    history.epoch,
                    history.history[metric],
                    color=colors[0],
                    label="Train",
                )
                plt.plot(
                    history.epoch,
                    history.history["val_" + metric],
                    color=colors[0],
                    linestyle="--",
                    label="Val",
                )
                plt.xlabel("Epoch")
                plt.ylabel(name)
                if metric == "loss":
                    plt.ylim([0, plt.ylim()[1]])
                elif metric == "auc":
                    plt.ylim([0.8, 1])
                else:
                    plt.ylim([0, 1])
                plt.legend()

        plot_metrics(history)
        plt.tight_layout()

        # Save plot
        plot_path = self.model_dir / "training_history_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {plot_path}")
        plt.show()

    def plot_best_trial_history(self, clf):
        """Plot training history from the best AutoKeras trial."""
        print("\n=== PLOTTING BEST TRIAL HISTORY ===")

        try:
            # Get the best trial
            best_trial = clf.tuner.oracle.get_best_trials(1)[0]
            print(f"Best trial ID: {best_trial.trial_id}")
            print(f"Best trial score: {best_trial.score:.4f}")

            # Use captured history if available
            if hasattr(self, "captured_history") and self.captured_history:
                print(
                    f"Using captured training history ({len(self.captured_history)} epochs)"
                )
                self.plot_captured_history(self.captured_history)
            elif hasattr(best_trial, "history") and best_trial.history is not None:
                print("Found training history for best trial!")
                self.plot_training_history(best_trial.history)
            else:
                print("No training history available")
                print("Falling back to TensorBoard logs for detailed metrics...")
                self.plot_from_tensorboard_logs()

        except Exception as e:
            print(f"Could not plot best trial history: {e}")
            print("Falling back to TensorBoard logs...")
            self.plot_from_tensorboard_logs()

    @staticmethod
    def _freeze_batchnorm(model):
        # Keep BN layers in inference mode to avoid moving-mean/var drift
        for layer in model.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    def retrain_and_plot_best_model(
        self,
        clf,
        train_features,
        train_labels,
        val_features,
        val_labels,
        class_weight,
        callbacks,
    ):
        """Retrain the exported best model and plot its training history."""
        print("\n=== RETRAINING BEST MODEL (STABLE) ===")
        try:
            best_model = clf.export_model()
            self._freeze_batchnorm(best_model)

            optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

            best_model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=self.METRICS,
            )

            ckpt_path = self.model_dir / "retrain_best_checkpoint.keras"
            cbs = [
                keras.callbacks.ModelCheckpoint(
                    filepath=str(ckpt_path),
                    monitor="val_prc",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_prc",
                    patience=10,
                    mode="max",
                    restore_best_weights=True,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_prc",
                    mode="max",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=1,
                ),
                callbacks[1],  # TensorBoard
                SaturationLogger(val_features, val_labels),
            ]

            print("Retraining best model with stability safeguards...")
            history = best_model.fit(
                train_features,
                train_labels,
                epochs=self.EPOCHS,
                batch_size=self.BATCH_SIZE,
                validation_data=(val_features, val_labels),
                callbacks=cbs,
                class_weight=class_weight,
                verbose=1,
                shuffle=True,
            )

            if os.path.exists(ckpt_path):
                print(f"Loading best weights from checkpoint: {ckpt_path}")
                best_model = keras.models.load_model(ckpt_path)

            self.plot_training_history(history)
            retrained_model_path = (
                self.model_dir / "retrained_best_model_combined.keras"
            )
            best_model.save(retrained_model_path)
            print(f"Retrained best model saved to: {retrained_model_path}")

            return best_model, history

        except Exception as e:
            print(f"Error during retraining: {e}")
            self.plot_from_tensorboard_logs()
            return None, None

    def plot_captured_history(self, captured_history):
        """Plot training history from captured callback data."""
        print("\n=== PLOTTING CAPTURED TRAINING HISTORY ===")

        if not captured_history:
            print("No captured history available")
            return

        # Convert captured history to plotable format
        epochs = list(range(len(captured_history)))
        metrics = ["loss", "prc", "precision", "recall"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("AutoKeras Training Metrics (Captured)", fontsize=16)

        for n, metric in enumerate(metrics):
            ax = axes[n // 2, n % 2]
            name = metric.replace("_", " ").capitalize()

            # Extract training metrics
            train_values = [
                epoch[metric] for epoch in captured_history if metric in epoch
            ]
            val_values = [
                epoch[f"val_{metric}"]
                for epoch in captured_history
                if f"val_{metric}" in epoch
            ]

            if train_values:
                ax.plot(
                    epochs[: len(train_values)],
                    train_values,
                    color=colors[0],
                    label="Train",
                    linewidth=2,
                )
            if val_values:
                ax.plot(
                    epochs[: len(val_values)],
                    val_values,
                    color=colors[0],
                    linestyle="--",
                    label="Val",
                    linewidth=2,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name)
            if metric == "loss":
                ax.set_ylim([0, ax.get_ylim()[1]])
            else:
                ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.model_dir / "captured_training_history_combined.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Captured training history plot saved to: {plot_path}")
        plt.show()

    def save_model_and_history(self, model, history):
        """Save the final model and training history."""
        print("\n=== SAVING MODEL AND HISTORY ===")

        # Save final model
        model_path = self.model_dir / "final_model"
        model.save(model_path)
        print(f"Final model saved to: {model_path}")

        # Save training history
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_file = self.model_dir / f"training_history_{timestamp}_combined.pkl"
        with open(history_file, "wb") as f:
            pickle.dump(history, f)
        print(f"Training history saved to: {history_file}")

    def evaluate_on_test(self, model, test_features, test_labels):
        """Evaluate the model on test data."""
        print("\n=== EVALUATING ON TEST DATA ===")

        # Evaluate model
        (
            test_loss,
            test_tp,
            test_fp,
            test_tn,
            test_fn,
            test_accuracy,
            test_precision,
            test_recall,
            test_auc,
            test_prc,
        ) = model.evaluate(test_features, test_labels, verbose=0)

        print("Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  PRC: {test_prc:.4f}")
        print(f"  True Positives: {test_tp}")
        print(f"  False Positives: {test_fp}")
        print(f"  True Negatives: {test_tn}")
        print(f"  False Negatives: {test_fn}")

        # Calculate additional metrics
        bomb_correct = test_tp
        bomb_total = test_tp + test_fn
        non_bomb_correct = test_tn
        non_bomb_total = test_tn + test_fp

        print("\nDetailed Results:")
        print(
            f"  Bombs correctly classified: {bomb_correct}/{bomb_total} ({100*bomb_correct/bomb_total:.1f}%)"
        )
        non_bomb_pct = 100 * non_bomb_correct / non_bomb_total
        print(
            f"  Non-bombs correctly classified: {non_bomb_correct}/{non_bomb_total} "
            f"({non_bomb_pct:.1f}%)"
        )

        return {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "auc": test_auc,
            "prc": test_prc,
            "tp": test_tp,
            "fp": test_fp,
            "tn": test_tn,
            "fn": test_fn,
        }

    def evaluate_autokeras_model(self, clf, test_features, test_labels):
        """Evaluate the AutoKeras model directly without retraining."""
        print("\n=== EVALUATING AUTOKERAS MODEL (BEFORE RETRAINING) ===")

        # Get the best model from AutoKeras
        best_model = clf.export_model()
        print("AutoKeras best model architecture:")
        print(best_model.summary())

        # Evaluate model
        (
            test_loss,
            test_tp,
            test_fp,
            test_tn,
            test_fn,
            test_accuracy,
            test_precision,
            test_recall,
            test_auc,
            test_prc,
        ) = best_model.evaluate(test_features, test_labels, verbose=0)

        print("AutoKeras Model Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  PRC: {test_prc:.4f}")
        print(f"  True Positives: {test_tp}")
        print(f"  False Positives: {test_fp}")
        print(f"  True Negatives: {test_tn}")
        print(f"  False Negatives: {test_fn}")

        # Calculate additional metrics
        bomb_correct = test_tp
        bomb_total = test_tp + test_fn
        non_bomb_correct = test_tn
        non_bomb_total = test_tn + test_fp

        print("\nDetailed Results:")
        print(
            f"  Bombs correctly classified: {bomb_correct}/{bomb_total} ({100*bomb_correct/bomb_total:.1f}%)"
        )
        print(
            f"  Non-bombs correctly classified: {non_bomb_correct}/{non_bomb_total} "
            f"({100*non_bomb_correct/non_bomb_total:.1f}%)"
        )

        return {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "auc": test_auc,
            "prc": test_prc,
            "tp": test_tp,
            "fp": test_fp,
            "tn": test_tn,
            "fn": test_fn,
        }

    def save_autokeras_model(self, clf):
        """Save the AutoKeras model directly without retraining."""
        print("\n=== SAVING AUTOKERAS MODEL ===")

        # Get the best model from AutoKeras
        best_model = clf.export_model()

        # Save the AutoKeras model
        model_path = self.model_dir / "autokeras_best_model_combined.keras"
        best_model.save(model_path)
        print(f"AutoKeras best model saved to: {model_path}")

        return best_model

    def plot_autokeras_metrics(self, clf):
        """Plot AutoKeras training metrics and trial results."""
        print("\n=== PLOTTING AUTOKERAS METRICS ===")

        try:
            # Get the best trial
            best_trial = clf.tuner.oracle.get_best_trials(1)[0]

            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("AutoKeras Training Metrics", fontsize=16)

            # AutoKeras only stores final values, not epoch-by-epoch history
            # Let's create a more useful visualization with available data
            print("AutoKeras stores final trial results, not epoch-by-epoch history")

            # Get all trials and their scores
            trials = clf.tuner.oracle.trials
            trial_scores = [trial.score for trial in trials if trial.score is not None]
            trial_ids = [i for i, trial in enumerate(trials) if trial.score is not None]

            if trial_scores:
                # Create a comprehensive summary plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle("AutoKeras Training Summary", fontsize=16)

                # Plot 1: Trial Performance
                axes[0, 0].plot(
                    trial_ids, trial_scores, "bo-", linewidth=2, markersize=8
                )
                axes[0, 0].set_title("Trial Performance (Val PRC)")
                axes[0, 0].set_xlabel("Trial Number")
                axes[0, 0].set_ylabel("Validation PRC Score")
                axes[0, 0].grid(True, alpha=0.3)

                # Highlight best trial
                best_idx = trial_scores.index(max(trial_scores))
                axes[0, 0].plot(
                    trial_ids[best_idx],
                    trial_scores[best_idx],
                    "ro",
                    markersize=12,
                    label=f"Best: {trial_scores[best_idx]:.4f}",
                )
                axes[0, 0].legend()

                # Plot 2: Trial Status Distribution
                status_counts = {}
                for trial in trials:
                    status = trial.status
                    status_counts[status] = status_counts.get(status, 0) + 1

                if status_counts:
                    statuses = list(status_counts.keys())
                    counts = list(status_counts.values())
                    axes[0, 1].bar(statuses, counts, color=["green", "red", "orange"])
                    axes[0, 1].set_title("Trial Status Distribution")
                    axes[0, 1].set_ylabel("Number of Trials")
                    for i, count in enumerate(counts):
                        axes[0, 1].text(i, count + 0.1, str(count), ha="center")

                # Plot 3: Best Trial Summary
                axes[1, 0].text(
                    0.1, 0.9, f"Best Trial ID: {best_trial.trial_id}", fontsize=12
                )
                axes[1, 0].text(
                    0.1, 0.8, f"Best Val PRC: {best_trial.score:.4f}", fontsize=12
                )
                axes[1, 0].text(0.1, 0.7, f"Total Trials: {len(trials)}", fontsize=12)
                axes[1, 0].text(0.1, 0.6, f"Status: {best_trial.status}", fontsize=12)
                axes[1, 0].text(
                    0.1, 0.5, f"Hyperparameters:", fontsize=12, weight="bold"
                )

                # Show some key hyperparameters if available
                if hasattr(best_trial, "hyperparameters"):
                    hp = best_trial.hyperparameters
                    y_pos = 0.4
                    for key, value in list(hp.values.items())[:5]:  # Show first 5
                        axes[1, 0].text(0.15, y_pos, f"{key}: {value}", fontsize=10)
                        y_pos -= 0.05

                axes[1, 0].set_title("Best Trial Details")
                axes[1, 0].axis("off")

                # Plot 4: Performance Comparison
                if len(trial_scores) > 1:
                    axes[1, 1].hist(
                        trial_scores,
                        bins=min(5, len(trial_scores)),
                        alpha=0.7,
                        color="skyblue",
                        edgecolor="black",
                    )
                    axes[1, 1].axvline(
                        max(trial_scores),
                        color="red",
                        linestyle="--",
                        label=f"Best: {max(trial_scores):.4f}",
                    )
                    axes[1, 1].axvline(
                        np.mean(trial_scores),
                        color="green",
                        linestyle="--",
                        label=f"Mean: {np.mean(trial_scores):.4f}",
                    )
                    axes[1, 1].set_title("Score Distribution")
                    axes[1, 1].set_xlabel("Validation PRC Score")
                    axes[1, 1].set_ylabel("Frequency")
                    axes[1, 1].legend()
                else:
                    axes[1, 1].text(
                        0.5,
                        0.5,
                        "Only one trial completed",
                        ha="center",
                        va="center",
                        transform=axes[1, 1].transAxes,
                    )
                    axes[1, 1].set_title("Score Distribution")
                    axes[1, 1].axis("off")

                plt.tight_layout()

                # Save plot
                plot_path = self.model_dir / "autokeras_training_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"AutoKeras training summary plot saved to: {plot_path}")
                plt.show()

            else:
                print("No trial scores available for plotting")

        except Exception as e:
            print(f"Could not plot AutoKeras metrics: {e}")
            print(
                "This is normal if AutoKeras doesn't provide detailed training history"
            )

        except Exception as e:
            print(f"Could not plot AutoKeras metrics: {e}")
            print(
                "This is normal if AutoKeras doesn't provide detailed training history"
            )

        # Try to plot from TensorBoard logs
        self.plot_from_tensorboard_logs()

    def plot_from_tensorboard_logs(self):
        """Plot metrics from TensorBoard logs."""
        print("\n=== PLOTTING FROM TENSORBOARD LOGS ===")

        try:
            # Find the most recent log directory
            log_dirs = list(self.logs_dir.glob("*"))
            if not log_dirs:
                print("No TensorBoard logs found")
                return

            # Get the most recent log directory
            latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Using logs from: {latest_log_dir}")

            # Read TensorBoard logs using tensorboard
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("AutoKeras Training Metrics (from TensorBoard)", fontsize=16)

            # Read training metrics
            train_log_path = latest_log_dir / "train"
            if train_log_path.exists():
                ea = EventAccumulator(str(train_log_path))
                ea.Reload()

                # Get available tags
                tags = ea.Tags()["scalars"]
                print(f"Available metrics: {tags}")

                # Plot loss
                if "loss" in tags:
                    loss_events = ea.Scalars("loss")
                    epochs = [event.step for event in loss_events]
                    values = [event.value for event in loss_events]
                    axes[0, 0].plot(
                        epochs, values, "b-", label="Train Loss", linewidth=2
                    )
                    axes[0, 0].set_title("Loss")
                    axes[0, 0].set_xlabel("Epoch")
                    axes[0, 0].set_ylabel("Loss")
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

                # Plot PRC
                if "prc" in tags:
                    prc_events = ea.Scalars("prc")
                    epochs = [event.step for event in prc_events]
                    values = [event.value for event in prc_events]
                    axes[0, 1].plot(
                        epochs, values, "g-", label="Train PRC", linewidth=2
                    )
                    axes[0, 1].set_title("Precision-Recall Curve (PRC)")
                    axes[0, 1].set_xlabel("Epoch")
                    axes[0, 1].set_ylabel("PRC")
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)

                # Plot accuracy
                if "accuracy" in tags:
                    acc_events = ea.Scalars("accuracy")
                    epochs = [event.step for event in acc_events]
                    values = [event.value for event in acc_events]
                    axes[1, 0].plot(
                        epochs, values, "r-", label="Train Accuracy", linewidth=2
                    )
                    axes[1, 0].set_title("Accuracy")
                    axes[1, 0].set_xlabel("Epoch")
                    axes[1, 0].set_ylabel("Accuracy")
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)

            # Read validation metrics
            val_log_path = latest_log_dir / "validation"
            if val_log_path.exists():
                ea = EventAccumulator(str(val_log_path))
                ea.Reload()

                tags = ea.Tags()["scalars"]
                print(f"Available validation metrics: {tags}")

                # Plot validation PRC
                if "prc" in tags:
                    prc_events = ea.Scalars("prc")
                    epochs = [event.step for event in prc_events]
                    values = [event.value for event in prc_events]
                    axes[1, 1].plot(epochs, values, "g--", label="Val PRC", linewidth=2)
                    axes[1, 1].set_title("Validation PRC")
                    axes[1, 1].set_xlabel("Epoch")
                    axes[1, 1].set_ylabel("PRC")
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = self.model_dir / "autokeras_tensorboard_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"TensorBoard metrics plot saved to: {plot_path}")
            plt.show()

        except Exception as e:
            print(f"Could not read TensorBoard logs: {e}")

    def create_metadata(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        class_weight,
        test_results,
    ):
        """Create metadata file documenting the training process."""
        print("\n=== CREATING METADATA ===")

        metadata_file = self.model_dir / "training_metadata.txt"

        with open(metadata_file, "w") as f:
            f.write("MODEL TRAINING METADATA\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("TRAINING PARAMETERS:\n")
            f.write(f"  Epochs: {self.EPOCHS}\n")
            f.write(
                f"  Batch size: {self.BATCH_SIZE} ({'GPU' if GPU_AVAILABLE else 'CPU'} optimized)\n"
            )
            f.write(
                f"  Max trials: {self.MAX_TRIALS} (increased for better architecture search)\n"
            )
            f.write(f"  Max model size: {self.MAX_MODEL_SIZE:,}\n")
            f.write(f"  Random seed: {self.seed}\n")
            f.write(f"  Hardware: {'GPU accelerated' if GPU_AVAILABLE else 'CPU'}\n\n")

            f.write("DATA STATISTICS:\n")
            f.write(f"  Training samples: {len(train_features):,}\n")
            f.write(f"  Training bombs: {np.sum(train_labels):,}\n")
            f.write(
                f"  Training non-bombs: {len(train_labels) - np.sum(train_labels):,}\n"
            )
            f.write(f"  Test samples: {len(test_features):,}\n")
            f.write(f"  Test bombs: {np.sum(test_labels):,}\n")
            f.write(f"  Test non-bombs: {len(test_labels) - np.sum(test_labels):,}\n\n")

            f.write("CLASS WEIGHTS:\n")
            f.write(f"  Class 0 (non-bomb): {class_weight[0]:.2f}\n")
            f.write(f"  Class 1 (bomb): {class_weight[1]:.2f}\n\n")

            f.write("TEST RESULTS:\n")
            f.write(f"  Loss: {test_results['loss']:.4f}\n")
            f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"  Precision: {test_results['precision']:.4f}\n")
            f.write(f"  Recall: {test_results['recall']:.4f}\n")
            f.write(f"  AUC: {test_results['auc']:.4f}\n")
            f.write(f"  PRC: {test_results['prc']:.4f}\n")
            f.write(f"  True Positives: {test_results['tp']}\n")
            f.write(f"  False Positives: {test_results['fp']}\n")
            f.write(f"  True Negatives: {test_results['tn']}\n")
            f.write(f"  False Negatives: {test_results['fn']}\n\n")

            f.write("NOTES:\n")
            f.write("  - Model uses AutoKeras for architecture search\n")
            f.write("  - Optimized for PRC (Precision-Recall Curve)\n")
            f.write("  - Uses augmented training data for better class balance\n")
            f.write("  - Early stopping with patience=10 on val_prc\n")
            f.write("  - Using AutoKeras model directly (no retraining)\n")
            f.write("  - AutoKeras handles learning rate optimization\n")
            f.write(f"  - {'GPU' if GPU_AVAILABLE else 'CPU'} accelerated training\n")

        print(f"Metadata saved to: {metadata_file}")

    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("Starting model training pipeline...")
        print("=" * 60)

        # Step 1: Load data
        train_features, train_labels, test_features, test_labels, input_shape = (
            self.load_data()
        )

        # Step 2: Calculate class weights
        class_weight = self.calculate_class_weights(train_labels)

        # Step 3: Split training data
        print("\n=== SPLITTING TRAINING DATA ===")
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features,
            train_labels,
            test_size=0.15,
            random_state=self.seed,
            stratify=train_labels,
        )
        print(f"Training samples: {len(train_features):,}")
        print(f"Validation samples: {len(val_features):,}")

        # Step 4: Setup callbacks
        early_stopping, tensorboard_callback = self.setup_callbacks()
        callbacks = (early_stopping, tensorboard_callback)

        # Step 5: Create and train AutoKeras model
        clf = self.create_autokeras_model(input_shape)
        clf, search_history = self.train_autokeras_model(
            clf,
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_weight,
            callbacks,
        )

        # Step 6: Evaluate AutoKeras model directly (before retraining)
        autokeras_results = self.evaluate_autokeras_model(
            clf, test_features, test_labels
        )

        # Step 7: Save AutoKeras model directly
        best_model = self.save_autokeras_model(clf)

        # Step 8: Retrain best model and plot its history
        retrained_model, retrain_history = self.retrain_and_plot_best_model(
            clf,
            train_features,
            train_labels,
            val_features,
            val_labels,
            class_weight,
            callbacks,
        )

        # Step 9: Re-evaluate on actual test data after retraining
        if retrained_model is not None:
            print("\n=== EVALUATING RETRAINED MODEL ON TEST DATA ===")
            test_results = self.evaluate_on_test(
                retrained_model, test_features, test_labels
            )
        else:
            # Fallback to AutoKeras results if retraining failed
            test_results = autokeras_results

        # Step 10: Create metadata
        self.create_metadata(
            train_features,
            train_labels,
            test_features,
            test_labels,
            class_weight,
            test_results,
        )

        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETE!")
        print(f"AutoKeras model saved in: {self.model_dir}")
        print(f"Training logs in: {self.logs_dir}")

        return (
            best_model,
            None,
            test_results,
        )  # search_history is None since we retrain separately


def main():
    """Run the model training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train bomb detection model using AutoKeras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O arguments
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Base data directory"
    )
    parser.add_argument(
        "--train-pickle",
        type=str,
        default="train_features_labels.pickle",
        help="Path to training features pickle file",
    )
    parser.add_argument(
        "--test-pickle",
        type=str,
        default="test_features_labels.pickle",
        help="Path to test features pickle file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="../models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="../logs",
        help="Directory to save training logs",
    )

    args = parser.parse_args()

    trainer = ModelTrainer(
        data_dir=args.data_dir,
        train_pickle=args.train_pickle,
        test_pickle=args.test_pickle,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
    )
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
