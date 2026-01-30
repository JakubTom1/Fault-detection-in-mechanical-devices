import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dropout


class MCDropout(Dropout):
    """
    Custom Dropout layer that is ALWAYS active, even during inference.
    Required for Monte Carlo Uncertainty Estimation.
    """

    def call(self, inputs):
        return super().call(inputs, training=True)


class TimeSeriesUtils:
    @staticmethod
    def load_data(data_dir, filename='time_series_data.npz', merge_train_val=False):
        """
        Loads time series data from .npz file.

        Args:
            merge_train_val (bool): If True, concatenates Train and Validation sets.
                                    Recommended for GNB/QDA/RF training to use more data.
        """
        train_path = os.path.join(data_dir, 'train.npz')
        val_path = os.path.join(data_dir, 'val.npz')
        test_path = os.path.join(data_dir, 'test.npz')
        train_data = np.load(train_path)
        val_data = np.load(val_path)
        test_data = np.load(test_path)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Data file not found at: {val_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Data file not found at: {test_path}")


        X_train = train_data['X']
        y_train = train_data['y']
        X_val = val_data['X']
        y_val = val_data['y']
        X_test = test_data['X']
        y_test = test_data['y']

        if merge_train_val:
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def flatten_windows(X):
        """
        Flattens 3D time windows [samples, time_steps, features]
        into 2D [samples, time_steps * features] for GNB/QDA.
        """
        return X.reshape(X.shape[0], -1)

    @staticmethod
    def predict_lstm_mc_uncertainty(model, X, n_iter=50):
        """
        Performs Monte Carlo Dropout prediction for LSTM.
        """
        print(f"Running {n_iter} MC simulations for LSTM...")
        # Stack predictions: [n_iter, n_samples, n_classes]
        preds = np.stack([model.predict(X, verbose=0) for _ in range(n_iter)])

        # Mean prediction (Point estimate)
        mean_proba = np.mean(preds, axis=0)

        # Epistemic uncertainty (Variance) across MC samples
        # Mean variance across all classes
        epistemic_var = np.mean(np.var(preds, axis=0), axis=1)
        epistemic_std = np.mean(np.std(preds, axis=0), axis=1)

        return mean_proba, epistemic_var, epistemic_std