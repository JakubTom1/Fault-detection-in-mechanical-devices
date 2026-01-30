import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def plot_uncertainty_intervals(y_true, probs, uncertainty_std, model_name, n_samples=56):
    """
    Visualizes predictions with uncertainty error bars (Epistemic Uncertainty).
    Mimics scientific plots.
    """
    indices = np.arange(len(y_true))[:n_samples]

    pred_class = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    colors = ['green' if p == t else 'red' for p, t in zip(pred_class[indices], y_true[indices])]

    plt.figure(figsize=(12, 6))

    plt.errorbar(
        x=indices,
        y=max_probs[indices],
        yerr=2 * uncertainty_std[indices],
        fmt='o',
        ecolor='gray',
        capsize=5,
        label='95% Confidence Interval (Epistemic)'
    )

    plt.scatter(indices, max_probs[indices], c=colors, s=50, zorder=3)

    plt.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.1)
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Probability")
    plt.title(f"{model_name}: Prediction Intervals & Correctness\n(Green=Correct, Red=Error, Bar=Model Uncertainty)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Incorrect'),
        Line2D([0], [0], color='gray', label='Uncertainty Range')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.show()

def evaluate_uncertainty_utility(y_true, probs, uncertainty_score, model_name):
    """
    Evaluates if uncertainty score can predict model errors (AUROC).
    High AUC means the model effectively signals when it is likely to be wrong.
    """
    preds = np.argmax(probs, axis=1)
    is_error = (preds != y_true).astype(int)

    if np.sum(is_error) == 0:
        print(f"{model_name}: No errors on test set (Perfect Accuracy).")
        return 1.0

    auc = roc_auc_score(is_error, uncertainty_score)
    print(f"{model_name} Error Detection AUC: {auc:.4f}")
    return auc
