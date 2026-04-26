# Privacy-Preserving Federated Learning for Intrusion Detection

Implementation of federated learning with differential privacy for network intrusion detection.

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
pip install numpy torch pandas scikit-learn matplotlib
```

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

### 2. Run Experiments
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

## Project Structure
```
project/
├── data_prep.py           # Data loading and preprocessing
├── model_training.py      # Neural network and training functions
├── main.py               # Main experiment runner
├── README.md            # This file
├── KDDTrain+.txt        #dataset
└── KDDTest+.txt         #dataset
```

## Author
Akash Dathaprasad (N01642373)  
University of North Florida  
CIS 6372: Information Assurance

## License
MIT License

## Citation

If you use this code, please cite:
```
Privacy-Preserving Federated Learning for Collaborative Network Intrusion Detection Systems
University of North Florida, 2026
```

## Dataset Citation

NSL-KDD Dataset:
```
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). 
A detailed analysis of the KDD CUP 99 data set. 
In IEEE Symposium on Computational Intelligence for Security and Defense Applications.
```
