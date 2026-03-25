# Privacy-Preserving Federated Learning for Intrusion Detection

Simple implementation of federated learning with differential privacy for network intrusion detection.

## Project Overview

This project demonstrates:
1. **Centralized Training** - Traditional ML approach (baseline)
2. **Federated Learning (Random Split)** - Collaborative training with uniform data distribution
3. **Federated Learning (Non-IID Split)** - Realistic scenario with heterogeneous data
4. **Federated Learning with Differential Privacy** - Maximum privacy protection

## Setup

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Download NSL-KDD Dataset

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

Place both files in the **same directory** as the Python scripts:
```
project-folder/
├── KDDTrain+.txt        ← Place here
├── KDDTest+.txt         ← Place here
├── data_prep.py
├── model_training.py
├── main.py
├── requirements.txt
└── README.md
```

### 3. Run Experiments
```bash
python main.py
```

## What the Code Does

### data_prep.py
- Loads NSL-KDD dataset
- Converts text features to numbers using label encoding
- Normalizes all values using standard scaling
- Splits data for 3 clients (random or non-IID)

### model_training.py
- Simple 3-layer neural network (64→32→2 neurons)
- Centralized training function
- Federated learning function
- Differential privacy noise addition
- Model aggregation using weighted averaging
- Evaluation metrics calculation

### main.py
- Runs all 4 experiments sequentially
- Compares results across approaches
- Creates visualization graphs

## Expected Results

- **Centralized**: ~93% accuracy (best performance, no privacy)
- **Federated Random**: ~91-92% accuracy (good privacy, minimal accuracy loss)
- **Federated Non-IID**: ~88-90% accuracy (realistic scenario with data heterogeneity)
- **Federated + DP**: ~85-88% accuracy (strongest privacy, measurable accuracy cost)

## Files Created After Running

- `results_comparison.png` - Accuracy comparison and learning progress
- `detailed_metrics.png` - Comprehensive metrics comparison

## How to Modify

### Change number of clients:
```python
client_data = split_data_for_clients(X_train, y_train, num_clients=5)  # Change 3 to 5
```

### Change privacy budget:
```python
dp_model, dp_acc, dp_time = train_federated(..., epsilon=1.0)  # Change 2.0 to 1.0
```

### Change training rounds:
```python
fl_model, fl_acc, fl_time = train_federated(..., rounds=50)  # Change 30 to 50
```

## Understanding the Output

The program will print:
1. Data loading and preprocessing progress
2. Training progress for each approach (every 5 epochs for centralized, every round for federated)
3. Final metrics (accuracy, precision, recall, F1-score) for each method
4. Comparison table of all approaches
5. Training time for each method

## Troubleshooting

**"File not found" error**: 
- Ensure `KDDTrain+.txt` and `KDDTest+.txt` are in the same folder as `main.py`
- Check file names are exact (with the `+` sign)

**"Out of memory" error**: 
- Reduce number of training rounds
- Close other applications

**Low accuracy**: 
- Ensure dataset files are correct
- Check if all required packages are installed
- Try increasing number of epochs/rounds

## Project Structure
```
project/
├── data_prep.py           # Data loading and preprocessing
├── model_training.py      # Neural network and training functions
├── main.py               # Main experiment runner
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── KDDTrain+.txt        # Dataset (not included - download separately)
└── KDDTest+.txt         # Dataset (not included - download separately)
```

## Author
Master's Student  
University of North Florida  
CIS 6372: Information Assurance

## License
MIT License

## Citation

If you use this code, please cite:
```
Privacy-Preserving Federated Learning for Collaborative Network Intrusion Detection Systems
University of North Florida, 2025
```

## Dataset Citation

NSL-KDD Dataset:
```
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). 
A detailed analysis of the KDD CUP 99 data set. 
In IEEE Symposium on Computational Intelligence for Security and Defense Applications.
```
