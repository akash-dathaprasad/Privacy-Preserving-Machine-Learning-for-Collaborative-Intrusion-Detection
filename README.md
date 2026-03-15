# Privacy-Preserving Federated Learning for Intrusion Detection

Simple implementation of federated learning with differential privacy for network intrusion detection.

## Project Overview

This project demonstrates:
1. **Centralized Training** - Traditional ML approach (baseline)
2. **Federated Learning** - Collaborative training without sharing data
3. **Non-IID Data** - Realistic scenario with different attack distributions
4. **Differential Privacy** - Adding noise for extra privacy protection


## Dataset Setup

### Download Dataset

The NSL-KDD dataset is **NOT included** in this repository. Please download it separately:

**Option 1: Kaggle (Recommended)**
1. Go to: https://www.kaggle.com/datasets/hassan06/nslkdd
2. Click "Download" (requires free Kaggle account)
3. Extract the ZIP file

**Option 2: GitHub Backup**
1. Go to: https://github.com/jmnwong/NSL-KDD-Dataset
2. Download the repository
3. Navigate to the dataset folder

### Required Files

You need these two files:
- `KDDTrain+.txt`
- `KDDTest+.txt`

### File Placement

Place both files in the **same directory** as `main.py`:
```
federated-learning-ids/
├── KDDTrain+.txt        ← Place here
├── KDDTest+.txt         ← Place here
├── data_prep.py
├── model_training.py
├── main.py
├── requirements.txt
└── README.md
```

### Verify Setup

After placing the files, verify by running:
```bash
python main.py
```

If you see "Loading data..." followed by sample counts, the setup is correct!

## What the Code Does

### data_prep.py
- Loads NSL-KDD dataset
- Converts text features to numbers
- Normalizes all values
- Splits data for 3 clients (random or non-IID)

### model_training.py
- Simple 3-layer neural network
- Centralized training function
- Federated learning function
- Differential privacy noise addition
- Model aggregation (averaging)

### main.py
- Runs all 4 experiments
- Compares results
- Creates comparison graphs

## Files Created

After running:
- `results_comparison.png` - Accuracy comparison chart
- `detailed_metrics.png` - All metrics comparison

## Expected Results

- **Centralized**: ~93% accuracy (best, but no privacy)
- **Federated Random**: ~91-92% accuracy (good privacy, similar performance)
- **Federated Non-IID**: ~88-90% accuracy (realistic scenario)
- **Federated + DP**: ~85-88% accuracy (strongest privacy, some accuracy loss)

## How to Modify

### Change number of clients:
In `main.py`, modify:
```python
client_data_random = split_data_for_clients(X_train, y_train, num_clients=5)  # Change 3 to 5
```

### Change privacy budget:
In `main.py`, modify:
```python
dp_model, dp_acc, dp_time = train_federated(..., epsilon=1.0)  # Change 2.0 to 1.0
```

### Change training rounds:
In `main.py`, modify:
```python
fl_model_random, fl_acc_random, fl_time_random = train_federated(..., rounds=50)  # Change 30 to 50
```

## Understanding the Output

The program will print:
1. Data loading progress
2. Training progress for each approach
3. Final accuracy for each method
4. Comparison table
5. Training times

## Troubleshooting

**"File not found" error**: 
- Make sure KDDTrain+.txt and KDDTest+.txt are in the same folder
- Update paths in main.py if needed

**"Out of memory" error**: 
- Reduce batch size in model_training.py
- Use fewer training rounds

**Poor accuracy**: 
- Increase number of epochs
- Adjust learning rate
- Check data preprocessing

## Author
Akash Dathaprasad - N01642373

