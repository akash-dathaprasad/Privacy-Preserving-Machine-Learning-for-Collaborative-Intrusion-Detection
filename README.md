# Privacy-Preserving Federated Learning for Intrusion Detection

Simple implementation of federated learning with differential privacy for network intrusion detection.

## Project Overview

This project demonstrates:
1. **Centralized Training** - Traditional ML approach (baseline)
2. **Federated Learning** - Collaborative training without sharing data
3. **Non-IID Data** - Realistic scenario with different attack distributions
4. **Differential Privacy** - Adding noise for extra privacy protection

## Setup

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Download NSL-KDD Dataset
Download from: https://www.unb.ca/cic/datasets/nsl.html

You need:
- KDDTrain+.txt
- KDDTest+.txt

Place them in the project folder.

### 3. Run Experiments
```bash
python main.py
```

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
Master's Student - University of North Florida
CIS 6372: Information Assurance

## License
MIT License
