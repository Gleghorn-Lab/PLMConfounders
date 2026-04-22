import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# Publication style
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SENSITIVITY = 0.965
SPECIFICITY = 0.682

np.random.seed(42)
n_samples = 100000
n_positive = n_samples // 2
n_negative = n_samples // 2

true_labels = np.concatenate([
    np.ones(n_positive, dtype=int),
    np.zeros(n_negative, dtype=int)
])

predictions = np.zeros_like(true_labels)

positive_indices = np.where(true_labels == 1)[0]
n_correct_positives = int(SENSITIVITY * len(positive_indices))
correct_positive_indices = np.random.choice(positive_indices, n_correct_positives, replace=False)
predictions[correct_positive_indices] = 1
incorrect_positive_indices = np.setdiff1d(positive_indices, correct_positive_indices)
predictions[incorrect_positive_indices] = 0

negative_indices = np.where(true_labels == 0)[0]
n_correct_negatives = int(SPECIFICITY * len(negative_indices))
correct_negative_indices = np.random.choice(negative_indices, n_correct_negatives, replace=False)
predictions[correct_negative_indices] = 0
incorrect_negative_indices = np.setdiff1d(negative_indices, correct_negative_indices)
predictions[incorrect_negative_indices] = 1

# Prediction scores for AUC
prediction_scores = np.zeros(len(true_labels), dtype=float)
prediction_scores[correct_positive_indices] = np.random.uniform(0.55, 1.0, len(correct_positive_indices))
prediction_scores[incorrect_positive_indices] = np.random.uniform(0.0, 0.45, len(incorrect_positive_indices))
prediction_scores[correct_negative_indices] = np.random.uniform(0.0, 0.45, len(correct_negative_indices))
prediction_scores[incorrect_negative_indices] = np.random.uniform(0.55, 1.0, len(incorrect_negative_indices))

print(f"Dataset size: {len(true_labels)}")
print(f"Positive samples: {np.sum(true_labels == 1)}")
print(f"Negative samples: {np.sum(true_labels == 0)}")
print(f"Predicted positive: {np.sum(predictions == 1)}")
print(f"Predicted negative: {np.sum(predictions == 0)}")
print()

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
auc = roc_auc_score(true_labels, prediction_scores)
mcc = matthews_corrcoef(true_labels, predictions)

cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)

print("Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (PPV): {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print()

print("Confusion Matrix:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
print()

print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))

# --- Publication-quality figure ---
COLOR_PRIMARY = '#2563EB'
COLOR_SECONDARY = '#F97316'
COLOR_DIAGONAL = '#94A3B8'
COLOR_CM_LOW = '#EFF6FF'
COLOR_CM_HIGH = '#1E40AF'

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# ROC curve
ax = axes[0]
fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
ax.fill_between(fpr, tpr, alpha=0.12, color=COLOR_PRIMARY)
ax.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2.2, label=f'ROC AUC = {auc:.3f}')
ax.plot([0, 1], [0, 1], color=COLOR_DIAGONAL, lw=1.2, linestyle='--', label='Random', zorder=0)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#D1D5DB')
ax.set_aspect('equal')

# Precision-Recall curve
ax = axes[1]
prec_curve, rec_curve, _ = precision_recall_curve(true_labels, prediction_scores)
pr_auc = np.abs(np.trapz(prec_curve, rec_curve))
ax.fill_between(rec_curve, prec_curve, alpha=0.12, color=COLOR_SECONDARY)
ax.plot(rec_curve, prec_curve, color=COLOR_SECONDARY, lw=2.2, label=f'PR AUC = {pr_auc:.3f}')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='#D1D5DB')

# Confusion matrix
ax = axes[2]
from matplotlib.colors import LinearSegmentedColormap
cm_cmap = LinearSegmentedColormap.from_list('cm_blue', [COLOR_CM_LOW, COLOR_CM_HIGH])
cm_norm = cm / cm.sum()
im = ax.imshow(cm_norm, interpolation='nearest', cmap=cm_cmap, vmin=0, vmax=cm_norm.max())

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Confusion Matrix')

for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = 100 * cm_norm[i, j]
        text_color = 'white' if cm_norm[i, j] > cm_norm.max() * 0.5 else '#1E293B'
        ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=text_color)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
for spine in ax.spines.values():
    spine.set_color('#D1D5DB')

plt.tight_layout(w_pad=3.0)
plt.savefig('accidental_taxonomist_results/base_cheating_scores.png', dpi=300, bbox_inches='tight')

print(f"\nVerification:")
print(f"Expected sensitivity: {SENSITIVITY:.1%} -> Actual: {sensitivity:.1%}")
print(f"Expected specificity: {SPECIFICITY:.1%} -> Actual: {specificity:.1%}")
