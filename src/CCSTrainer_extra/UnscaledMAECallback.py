import keras
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class UnscaledMAECallback(keras.callbacks.Callback):
    """
    Keras callback that tracks and logs training metrics in original (unscaled) units.

    Extends Keras Callback to convert model predictions from scaled back to original
    target units, enabling evaluation in interpretable values (e.g., actual CCS or mobility).
    Automatically logs unscaled MAE/MSE to CSV and generates training curve visualizations.
    """

    def __init__(self, X_train, adducts_train_encoded, X_val, adducts_val_encoded,
                 y_train_scaled, y_val_scaled, y_scaler, fold_dir):
        """
        Initialize the unscaled metrics callback with training data and scaler.

        Args:
            X_train (array): Training feature matrix
            adducts_train_encoded (array): One-hot encoded adducts for training set
            X_val (array): Validation feature matrix
            adducts_val_encoded (array): One-hot encoded adducts for validation set
            y_train_scaled (array): Scaled training target values
            y_val_scaled (array): Scaled validation target values
            y_scaler (object): Fitted scaler with inverse_transform method
            fold_dir (str): Directory path for saving logs and plots

        Returns:
            UnscaledMAECallback: Initialized callback instance
        """

        super().__init__()
        self.X_train = X_train
        self.X_train_adducts = adducts_train_encoded
        self.X_val = X_val
        self.X_val_adducts = adducts_val_encoded
        self.y_train_scaled = y_train_scaled
        self.y_val_scaled = y_val_scaled
        self.y_scaler = y_scaler
        self.fold_dir = fold_dir

        # Precompute unscaled true values for BOTH training and validation
        self.y_train_unscaled = self.y_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
        self.y_val_unscaled = self.y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

        self.train_losses_unscaled = []
        self.val_losses_unscaled = []

        # CSV logging setup
        self.csv_path = os.path.join(fold_dir, "unscaled_logs.csv")
        self.csv_exists = os.path.exists(self.csv_path)

        # Store reference to the optimizer (will be set in on_train_begin and updated in on_epoch_begin)
        self.optimizer = None
        self.current_lr = None

        # Compute scale factor for unscaling deltas (works for linear scalers like StandardScaler or MinMaxScaler)
        self.scale_factor = (self.y_scaler.inverse_transform(np.array([[1]]))[0][0] -
                             self.y_scaler.inverse_transform(np.array([[0]]))[0][0])

    def on_train_begin(self, logs=None):
        """
        Setup callback at the beginning of training.

        Args:
            logs (dict, optional): Dictionary of logs passed by Keras
        """

        # Get the optimizer from the model
        self.optimizer = self.model.optimizer

    def on_epoch_begin(self, epoch, logs=None):
        """
        Record current learning rate at the start of each epoch.

        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs passed by Keras
        """
        self.current_lr = self.optimizer.learning_rate.numpy()

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate, log, and store unscaled metrics at epoch end.

        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs containing scaled metrics

        Side effects:
            - Appends unscaled metrics to CSV file
            - Updates internal tracking arrays
            - Adds unscaled metrics to logs dictionary
        """

        logs = logs or {}

        # Unscale MAEs from logs (train is training-mode, val is eval-mode)
        loss_unscaled = logs['loss'] * self.scale_factor  # Unscaled MAE on training set
        val_loss_unscaled = logs['val_loss'] * self.scale_factor  # Unscaled MAE on validation set

        # Unscale MSEs from logs (assuming 'mse' is in model metrics)
        mse_unscaled = logs['mse'] * (self.scale_factor ** 2)  # Unscaled MSE on training set
        val_mse_unscaled = logs['val_mse'] * (self.scale_factor ** 2)  # Unscaled MSE on validation set

        self.train_losses_unscaled.append(loss_unscaled)
        self.val_losses_unscaled.append(val_loss_unscaled)

        logs['val_mae_unscaled'] = val_loss_unscaled
        logs['train_mae_unscaled'] = loss_unscaled

        # Prepare row for CSV with ONLY the specified columns (all unscaled)
        row_data = {
            'epoch': epoch,
            'learning_rate': self.current_lr,
            'loss': loss_unscaled,  # Unscaled MAE on training set
            'mse': mse_unscaled,  # Unscaled MSE on training set
            'val_loss': val_loss_unscaled,  # Unscaled MAE on validation set
            'val_mse': val_mse_unscaled  # Unscaled MSE on validation set
        }

        # Write to CSV (append mode)
        df_new = pd.DataFrame([row_data])
        if self.csv_exists:
            df_new.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(self.csv_path, mode='w', header=True, index=False)
            self.csv_exists = True

    def on_train_end(self, logs=None):
        """
        Generate and save training curves visualization.

        Args:
            logs (dict, optional): Dictionary of logs passed by Keras

        Side effects:
            - Saves PNG plot of unscaled MAE curves to fold directory
        """

        # Plot with better styling to ensure curves are visible
        plt.figure()
        plt.plot(self.train_losses_unscaled, label='Train MAE')
        plt.plot(self.val_losses_unscaled, label='Val MAE')
        plt.title("Unscaled loss curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.fold_dir, "unscaled_loss_curve.png"))
        plt.close()
