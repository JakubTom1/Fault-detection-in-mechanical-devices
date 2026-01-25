import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

class UncertaintyEvaluator:
    """
    Unified class for uncertainty evaluation across models:
    - Scikit-learn (GNB, QDA, RF)
    - XGBoost
    - PyMC (Bayesian models)
    """

    def __init__(self, y_true, model_name):
        # Ground-truth labels
        self.y_true = y_true
        # Human-readable model identifier
        self.model_name = model_name
        # Container for computed metrics
        self.results = {}

    def compute_metrics(self, probs, epistemic_var=None):
        """
        Computes a comprehensive set of uncertainty-related metrics.

        Parameters
        ----------
        probs : ndarray of shape [n_samples, n_classes]
            Predicted class probabilities.
        epistemic_var : ndarray of shape [n_samples], optional
            Per-sample epistemic uncertainty (e.g., RF tree variance or Bayesian posterior variance).
        """
        # 1. Point predictions and confidence scores
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracy = np.mean(preds == self.y_true)

        # 2. Proper scoring rules (Negative Log-Likelihood, Brier Score)
        # Numerical stability for logarithms
        epsilon = 1e-15
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

        # Multiclass negative log-likelihood
        nll = log_loss(self.y_true, probs_clipped)

        # Multiclass Brier score (squared error in probability space)
        y_ohe = np.eye(probs.shape[1])[self.y_true]
        brier = np.mean(np.sum((probs - y_ohe) ** 2, axis=1))

        # 3. Predictive entropy (total uncertainty)
        # H(p) = -sum(p * log(p))
        total_entropy = entropy(probs_clipped, axis=1)
        avg_entropy = np.mean(total_entropy)

        # 4. Expected Calibration Error (ECE)
        ece = self._compute_ece(self.y_true, preds, confidences)

        # Store aggregated results
        self.results = {
            "Model": self.model_name,
            "Accuracy": accuracy,
            "NLL": nll,
            "Brier Score": brier,
            "ECE": ece,
            "Avg Entropy (Total Uncertainty)": avg_entropy,
            "Avg Epistemic Var": np.mean(epistemic_var) if epistemic_var is not None else np.nan
        }

        return self.results, total_entropy, epistemic_var

    def _compute_ece(self, y_true, y_pred, confidences, n_bins=10):
        """
        Computes Expected Calibration Error (ECE).

        Measures the mismatch between predicted confidence and empirical accuracy.
        Critical for detecting overconfident or underconfident models.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Identify samples whose confidence falls into the current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    @staticmethod
    def extract_rf_uncertainty(model, X):
        """
        Random Forest–specific uncertainty extraction.

        Returns
        -------
        avg_proba : ndarray [n_samples, n_classes]
            Mean predicted probabilities across all trees.
        variance : ndarray [n_samples]
            Variance across trees (epistemic uncertainty proxy).
        """
        # Collect per-tree probability predictions
        # Shape: [n_trees, n_samples, n_classes]
        all_tree_preds = np.stack([tree.predict_proba(X) for tree in model.estimators_])

        # Standard Random Forest output: mean probability
        avg_proba = np.mean(all_tree_preds, axis=0)

        # Epistemic uncertainty: disagreement between trees
        # First variance across trees per class, then mean across classes
        variance = np.mean(np.var(all_tree_preds, axis=0), axis=1)

        return avg_proba, variance

    @staticmethod
    def extract_bayesian_uncertainty(trace, X_scaled, y_map):
        """
        Uncertainty extraction for Bayesian models (PyMC).

        Uses posterior samples to compute:
        - Bayesian model averaged predictions
        - Epistemic uncertainty via posterior predictive variance
        """
        import xarray as xr

        # Extract posterior samples of weights and biases
        # W: [chains, draws, features, classes]
        # b: [chains, draws, classes]
        W_post = trace.posterior["W"].values
        b_post = trace.posterior["b"].values

        # Flatten chains and draws into a single posterior sample dimension
        W_flat = W_post.reshape(-1, W_post.shape[2], W_post.shape[3])
        b_flat = b_post.reshape(-1, b_post.shape[2])

        n_samples = W_flat.shape[0]
        probs_list = []

        # Monte Carlo posterior predictive sampling
        # For each posterior draw, compute class probabilities
        for i in range(n_samples):
            logits = np.dot(X_scaled, W_flat[i]) + b_flat[i]

            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs_list.append(probs)

        # Shape: [n_posterior_samples, n_test_samples, n_classes]
        probs_stack = np.stack(probs_list)

        # Bayesian model averaging (mean predictive distribution)
        mean_proba = np.mean(probs_stack, axis=0)

        # Epistemic uncertainty: variance of posterior predictive probabilities
        epistemic_var = np.mean(np.var(probs_stack, axis=0), axis=1)

        return mean_proba, epistemic_var
