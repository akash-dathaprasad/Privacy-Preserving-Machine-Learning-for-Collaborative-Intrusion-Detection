"""
Main script to run all experiments
"""

import numpy as np
import torch
from data_prep import load_and_prepare_data, split_data_for_clients
from model_training import train_centralized, train_federated, calculate_metrics
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

print("="*80)
print("PRIVACY-PRESERVING FEDERATED LEARNING FOR INTRUSION DETECTION")
print("="*80)

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\nSTEP 1: Loading and preparing data...")

# Update these paths to your NSL-KDD dataset location
train_file = 'KDDTrain+.txt'
test_file = 'KDDTest+.txt'

X_train, X_test, y_train, y_test = load_and_prepare_data(train_file, test_file)

# ============================================================================
# STEP 2: Centralized Training (Baseline)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Centralized Training (Baseline)")
print("="*80)

cent_model, cent_accuracy, cent_time = train_centralized(
    X_train, y_train, X_test, y_test, epochs=20
)

cent_metrics = calculate_metrics(cent_model, X_test, y_test)
print(f"\nCentralized Results:")
print(f"  Accuracy:  {cent_metrics['accuracy']:.2f}%")
print(f"  Precision: {cent_metrics['precision']:.2f}%")
print(f"  Recall:    {cent_metrics['recall']:.2f}%")
print(f"  F1-Score:  {cent_metrics['f1_score']:.2f}%")

# ============================================================================
# STEP 3: Federated Learning - Random Split
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Federated Learning - Random Split")
print("="*80)

client_data_random = split_data_for_clients(X_train, y_train, num_clients=3, method='random')

fl_model_random, fl_acc_random, fl_time_random = train_federated(
    client_data_random, X_test, y_test, rounds=30, local_epochs=5, use_dp=False
)

fl_metrics_random = calculate_metrics(fl_model_random, X_test, y_test)
print(f"\nFederated Learning (Random) Results:")
print(f"  Accuracy:  {fl_metrics_random['accuracy']:.2f}%")
print(f"  Precision: {fl_metrics_random['precision']:.2f}%")
print(f"  Recall:    {fl_metrics_random['recall']:.2f}%")
print(f"  F1-Score:  {fl_metrics_random['f1_score']:.2f}%")

# ============================================================================
# STEP 4: Federated Learning - Non-IID Split
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Federated Learning - Non-IID Split")
print("="*80)

client_data_noniid = split_data_for_clients(X_train, y_train, num_clients=3, method='noniid')

fl_model_noniid, fl_acc_noniid, fl_time_noniid = train_federated(
    client_data_noniid, X_test, y_test, rounds=35, local_epochs=5, use_dp=False
)

fl_metrics_noniid = calculate_metrics(fl_model_noniid, X_test, y_test)
print(f"\nFederated Learning (Non-IID) Results:")
print(f"  Accuracy:  {fl_metrics_noniid['accuracy']:.2f}%")
print(f"  Precision: {fl_metrics_noniid['precision']:.2f}%")
print(f"  Recall:    {fl_metrics_noniid['recall']:.2f}%")
print(f"  F1-Score:  {fl_metrics_noniid['f1_score']:.2f}%")

# ============================================================================
# STEP 5: Federated Learning with Differential Privacy
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Federated Learning with Differential Privacy")
print("="*80)

dp_model, dp_acc, dp_time = train_federated(
    client_data_random, X_test, y_test, rounds=30, local_epochs=5, 
    use_dp=True, epsilon=2.0
)

dp_metrics = calculate_metrics(dp_model, X_test, y_test)
print(f"\nFederated Learning with DP Results:")
print(f"  Accuracy:  {dp_metrics['accuracy']:.2f}%")
print(f"  Precision: {dp_metrics['precision']:.2f}%")
print(f"  Recall:    {dp_metrics['recall']:.2f}%")
print(f"  F1-Score:  {dp_metrics['f1_score']:.2f}%")

# ============================================================================
# STEP 6: Summary and Comparison
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - COMPARISON OF ALL APPROACHES")
print("="*80)

print("\n{:<30} {:<12} {:<12} {:<12} {:<12}".format(
    "Approach", "Accuracy", "Precision", "Recall", "F1-Score"
))
print("-" * 80)

approaches = [
    ("Centralized", cent_metrics),
    ("Federated (Random)", fl_metrics_random),
    ("Federated (Non-IID)", fl_metrics_noniid),
    ("Federated + DP", dp_metrics)
]

for name, metrics in approaches:
    print("{:<30} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}".format(
        name,
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ))

print("\nTraining Times:")
print(f"  Centralized:          {cent_time:.2f} seconds")
print(f"  Federated (Random):   {fl_time_random:.2f} seconds")
print(f"  Federated (Non-IID):  {fl_time_noniid:.2f} seconds")
print(f"  Federated + DP:       {dp_time:.2f} seconds")

# ============================================================================
# STEP 7: Visualizations
# ============================================================================
print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

# Plot 1: Accuracy Comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart comparison
approaches_names = ["Centralized", "FL Random", "FL Non-IID", "FL + DP"]
accuracies = [
    cent_metrics['accuracy'],
    fl_metrics_random['accuracy'],
    fl_metrics_noniid['accuracy'],
    dp_metrics['accuracy']
]

axes[0].bar(approaches_names, accuracies, color=['blue', 'green', 'orange', 'red'])
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Accuracy Comparison')
axes[0].set_ylim([0, 100])
axes[0].grid(True, axis='y')

for i, v in enumerate(accuracies):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# Federated learning progress
axes[1].plot(range(1, len(fl_acc_random) + 1), fl_acc_random, marker='o', label='FL Random')
axes[1].plot(range(1, len(fl_acc_noniid) + 1), fl_acc_noniid, marker='s', label='FL Non-IID')
axes[1].plot(range(1, len(dp_acc) + 1), dp_acc, marker='^', label='FL + DP')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Federated Learning Progress')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300)
print("Saved: results_comparison.png")

# Plot 2: Detailed Metrics Comparison
fig, ax = plt.subplots(figsize=(12, 6))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
width = 0.2

for i, (name, metrics) in enumerate(approaches):
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    ax.bar(x + i*width, values, width, label=name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score (%)')
ax.set_title('Detailed Metrics Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, axis='y')
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('detailed_metrics.png', dpi=300)
print("Saved: detailed_metrics.png")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nResults saved:")
print("  - results_comparison.png")
print("  - detailed_metrics.png")
