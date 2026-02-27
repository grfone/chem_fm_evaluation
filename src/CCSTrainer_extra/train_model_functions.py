import csv
import os

import joblib

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error, \
    median_absolute_error
from sktime.performance_metrics.forecasting import mean_squared_percentage_error, median_absolute_percentage_error
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras import Input, Model
from keras.layers import Concatenate, Dense
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger

from src.CCSTrainer_extra.UnscaledMAECallback import UnscaledMAECallback


def train_a_fold(train_data, val_data, test_data, adduct_encoder, y_scaler, fold, fold_dir, nn):
    # Unpack data
    X_train, adducts_train_encoded, y_train_scaled = train_data
    X_val, adducts_val_encoded, y_val_scaled = val_data
    X_test, adducts_test_encoded, y_test_scaled = test_data

    # Build the model
    fingerprint_shape = (X_train.shape[1],)
    adduct_shape = (adducts_train_encoded.shape[1],)
    if nn == "singleDense":
        model = _build_singleDense(fingerprint_shape, adduct_shape)
    else:
        model = _build_simpleModel(fingerprint_shape, adduct_shape)

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

    # Train (fit the model)
    os.makedirs(fold_dir, exist_ok=True)  # Create the folders as needed
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, min_delta=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=30, min_lr=5e-6, min_delta=0.001)
    csv_logger = CSVLogger(os.path.join(fold_dir, 'log.csv'), append=True)
    unscaled_mae_callback = UnscaledMAECallback(
        X_train, adducts_train_encoded, X_val, adducts_val_encoded,
        y_train_scaled, y_val_scaled, y_scaler, fold_dir  # Pass the raw/unscaled y values
    )
    history = model.fit([X_train, adducts_train_encoded], y_train_scaled,
                        validation_data=([X_val, adducts_val_encoded], y_val_scaled),
                        epochs=10000, batch_size=32, callbacks=[early_stopping, reduce_lr, csv_logger, unscaled_mae_callback], verbose=1)

    # Save the model, the y_scaler and the onehot_encoder
    model.save(os.path.join(fold_dir, f'model.keras'))
    joblib.dump(y_scaler, os.path.join(fold_dir, f'y_scaler.pkl'))
    joblib.dump(adduct_encoder, os.path.join(fold_dir, f'adduct_encoder.pkl'))

    # Plot the loss curves
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(fold_dir, f"loss_curve.png"))
    plt.close()

    # Evaluate on test set
    train_data = y_train_scaled, X_train, adducts_train_encoded
    val_data = y_val_scaled, X_val, adducts_val_encoded
    test_data = y_test_scaled, X_test, adducts_test_encoded
    _evaluate_fold(train_data, val_data, test_data, model, y_scaler, fold, fold_dir)


def _build_singleDense(fingerprint_shape, adduct_shape):
    """Build the neural network model known as singleDense"""
    print(f"    ◦ Building singleDense model")
    fingerprints_input = Input(shape=fingerprint_shape, name="fingerprints_input")
    adducts_input = Input(shape=adduct_shape, name="adducts_input")

    x = fingerprints_input
    # x = BatchNormalization()(x)
    features = Concatenate()([x, adducts_input])
    output = Dense(1, activation="linear", name="ccs_output")(features)

    model = Model(inputs=[fingerprints_input, adducts_input], outputs=output, name= "singleDense")

    return model


def _build_simpleModel(fingerprint_shape, adduct_shape):
    """Build the neural network model known as simpleModel"""
    print(f"    ◦ Building simpleModel")
    fingerprints_input = Input(shape=fingerprint_shape, name="fingerprints_input")
    adducts_input = Input(shape=adduct_shape, name="adducts_input")

    number_of_neurons = 512
    x = fingerprints_input
    # x = BatchNormalization()(x)
    features = Concatenate()([x, adducts_input])
    x = Dense(number_of_neurons, activation='linear')(features)
    for _ in range(3):
        x = Dense(number_of_neurons, activation='relu', kernel_initializer='he_normal')(x)  # (Optional) , kernel_regularizer=L2(0.001)
    output = Dense(1, activation="linear", name="ccs_output")(x)

    model = Model(inputs=[fingerprints_input, adducts_input], outputs=output, name="simpleModel")

    return model


def _evaluate_fold(train_data, val_data, test_data, model, y_scaler, fold, fold_dir):
    print("    ◦ Evaluating fold")
    # --- Helpers --------------------------------------------------------------
    def plot_scatter(y_true, y_pred, y_pred_lin, r2, save_path, title_prefix=""):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, s=1)
        plt.plot(y_true, y_pred_lin, color='green', label=f'Linear fit (R² = {r2:.3f})')
        plt.title(f'{title_prefix} Scatter plot, R² = {r2:.3f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_scatter_adducts(y_true, y_pred, y_pred_lin, r2, adducts, save_path, title_prefix=""):
        first_three = np.any(adducts[:, :3], axis=1)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true[first_three], y_pred[first_three], alpha=0.5, s=1, color='red', label='Dimers')
        plt.scatter(y_true[~first_three], y_pred[~first_three], alpha=0.5, s=1, color='blue', label='Monomers')
        plt.plot(y_true, y_pred_lin, color='green', label=f'Linear fit (R² = {r2:.3f})')
        plt.title(f'{title_prefix} Scatter plot, R² = {r2:.3f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def process_split(name, X, adducts, y_scaled, out_dir):
        pred_scaled = model.predict([X, adducts], batch_size=32, verbose=0).flatten()
        y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        linreg = LinearRegression()
        linreg.fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
        y_pred_lin = linreg.predict(y_true.reshape(-1, 1)).flatten()
        r2 = r2_score(y_pred, y_pred_lin)
        # Normal scatter plot
        plot_scatter(y_true, y_pred, y_pred_lin, r2, save_path=os.path.join(out_dir, f"{name}_scatter.png"), title_prefix=f"{name.capitalize()} set")
        return y_true, y_pred, r2, y_pred_lin

    # --- Unpack data ----------------------------------------------------------
    y_train_scaled, X_train, adducts_train_encoded = train_data
    y_val_scaled, X_val, adducts_val_encoded = val_data
    y_test_scaled, X_test, adducts_test_encoded = test_data

    # --- Process train/val/test -----------------------------------------------
    y_train_true, y_train_pred, r2_train, y_train_lin = process_split("train", X_train, adducts_train_encoded, y_train_scaled, fold_dir)
    y_val_true, y_val_pred, r2_val, y_val_lin = process_split("val", X_val, adducts_val_encoded, y_val_scaled, fold_dir)
    y_test_true, y_test_pred, r2_test, y_test_lin = process_split("test", X_test, adducts_test_encoded, y_test_scaled, fold_dir)

    # --- Adduct-colored scatter plots ----------------------------------------
    plot_scatter_adducts(y_train_true, y_train_pred, y_train_lin, r2_train, adducts_train_encoded,
        save_path=os.path.join(fold_dir, 'train_scatter_adducts.png'), title_prefix="Train set")
    plot_scatter_adducts(y_val_true, y_val_pred, y_val_lin, r2_val, adducts_val_encoded,
        save_path=os.path.join(fold_dir, 'val_scatter_adducts.png'), title_prefix="Validation set")
    plot_scatter_adducts(y_test_true, y_test_pred, y_test_lin, r2_test, adducts_test_encoded,
        save_path=os.path.join(fold_dir, 'test_scatter_adducts.png'), title_prefix="Test set")

    # Predictions and test unscaled
    pred_scaled = model.predict([X_test, adducts_test_encoded], batch_size=32, verbose=0).flatten()
    y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Linear regression between y_test and y_pred
    linreg = LinearRegression()
    linreg.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
    y_pred_lin = linreg.predict(y_test.reshape(-1, 1)).flatten()

    # Calculate all th metrics
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_pred, y_pred_lin)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mspe = mean_squared_percentage_error(y_test, y_pred)
    medape = median_absolute_percentage_error(y_test, y_pred)

    # Save the metrics
    eval_path = os.path.join(f"{fold_dir[:-6]}_results.csv")
    write_header = not os.path.exists(eval_path)  # In the first fold the header is needed, not in the rest
    with open(eval_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write the header (if necessary)
        if write_header:
            writer.writerow([
                "Fold",
                "MAE",
                "MedAE",
                "MSE",
                "R²",
                "MAPE(%)",
                "MSPE(%)",
                "MedAPE(%)"
            ])
        # Write the metrics
        writer.writerow([
            f"{fold+1}",
            f"{mae:.4f}",
            f"{medae:.4f}",
            f"{mse:.4f}",
            f"{r2:.4f}",
            f"{mape:.4f}",
            f"{mspe:.4f}",
            f"{medape:.4f}"
        ])
