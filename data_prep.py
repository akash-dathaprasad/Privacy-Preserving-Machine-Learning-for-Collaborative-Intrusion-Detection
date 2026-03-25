"""
Simple data preprocessing for NSL-KDD dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_prepare_data(train_file, test_file):
    """
    Load and prepare NSL-KDD dataset
    
    Args:
        train_file: Path to training file
        test_file: Path to test file
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Column names for NSL-KDD dataset
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
              'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
              'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
              'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
              'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate',
              'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
              'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
              'dst_host_srv_count', 'dst_host_same_srv_rate',
              'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
              'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
              'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
              'dst_host_srv_rerror_rate', 'label', 'difficulty']
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_file, names=columns)
    test_df = pd.read_csv(test_file, names=columns)
    
    # Remove difficulty column
    train_df = train_df.drop('difficulty', axis=1)
    test_df = test_df.drop('difficulty', axis=1)
    
    # Convert labels to binary (0 = normal, 1 = attack)
    print("Converting labels...")
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Separate features and labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)
    
    # Handle categorical columns - simple label encoding
    print("Encoding categorical features...")
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        # Handle unknown categories in test set
        X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test[col] = le.transform(X_test[col])
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def split_data_for_clients(X, y, num_clients=3, method='random'):
    """
    Split data across clients
    
    Args:
        X: Features
        y: Labels
        num_clients: Number of clients
        method: 'random' or 'noniid'
        
    Returns:
        List of (X, y) tuples for each client
    """
    print(f"Splitting data for {num_clients} clients using {method} method...")
    
    if method == 'random':
        # Random split
        indices = np.random.permutation(len(X))
        splits = np.array_split(indices, num_clients)
        
    else:  # non-iid
        # Sort by labels to create non-iid distribution
        sorted_idx = np.argsort(y)
        splits = np.array_split(sorted_idx, num_clients)
    
    client_data = []
    for i, idx in enumerate(splits):
        client_data.append((X[idx], y[idx]))
        print(f"  Client {i}: {len(idx)} samples, {np.sum(y[idx])} attacks ({np.mean(y[idx])*100:.1f}%)")
    
    return client_data
