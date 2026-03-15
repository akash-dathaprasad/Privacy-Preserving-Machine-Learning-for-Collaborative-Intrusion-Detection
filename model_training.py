"""
Simple training functions for centralized and federated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time

# Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes: normal and attack
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_centralized(X_train, y_train, X_test, y_test, epochs=20):
    """
    Train model in centralized manner
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        
    Returns:
        model, accuracy, training_time
    """
    print("\n=== CENTRALIZED TRAINING ===")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create model
    model = SimpleNet(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train_t).sum().item() / len(y_train_t) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    
    training_time = time.time() - start_time
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test_t).sum().item() / len(y_test_t) * 100
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return model, test_accuracy, training_time


def train_federated(client_data, X_test, y_test, rounds=30, local_epochs=5, use_dp=False, epsilon=2.0):
    """
    Train model using federated learning
    
    Args:
        client_data: List of (X, y) tuples for each client
        X_test, y_test: Test data
        rounds: Number of federated rounds
        local_epochs: Epochs per client per round
        use_dp: Whether to use differential privacy
        epsilon: Privacy budget (if use_dp=True)
        
    Returns:
        global_model, accuracies, training_time
    """
    method = "FEDERATED LEARNING WITH DP" if use_dp else "FEDERATED LEARNING"
    print(f"\n=== {method} ===")
    
    # Convert test data to tensors
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create global model
    input_size = client_data[0][0].shape[1]
    global_model = SimpleNet(input_size)
    
    # Store accuracies
    round_accuracies = []
    
    start_time = time.time()
    
    for round_num in range(rounds):
        print(f"\nRound {round_num + 1}/{rounds}")
        
        # Store client models
        client_models = []
        client_sizes = []
        
        # Each client trains locally
        for client_id, (X_client, y_client) in enumerate(client_data):
            # Convert to tensors
            X_client_t = torch.FloatTensor(X_client)
            y_client_t = torch.LongTensor(y_client)
            
            # Create client model (copy of global model)
            client_model = SimpleNet(input_size)
            client_model.load_state_dict(global_model.state_dict())
            
            # Train locally
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(client_model.parameters(), lr=0.001)
            
            client_model.train()
            for epoch in range(local_epochs):
                optimizer.zero_grad()
                outputs = client_model(X_client_t)
                loss = criterion(outputs, y_client_t)
                loss.backward()
                
                # Add noise for differential privacy
                if use_dp:
                    add_noise_to_gradients(client_model, epsilon)
                
                optimizer.step()
            
            client_models.append(client_model)
            client_sizes.append(len(X_client))
        
        # Aggregate models (weighted average)
        global_model = aggregate_models(client_models, client_sizes)
        
        # Evaluate global model
        global_model.eval()
        with torch.no_grad():
            test_outputs = global_model(X_test_t)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test_t).sum().item() / len(y_test_t) * 100
        
        round_accuracies.append(test_accuracy)
        print(f"Global model accuracy: {test_accuracy:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final Test Accuracy: {round_accuracies[-1]:.2f}%")
    
    return global_model, round_accuracies, training_time


def aggregate_models(client_models, client_sizes):
    """
    Aggregate client models using weighted averaging
    
    Args:
        client_models: List of client models
        client_sizes: List of client dataset sizes
        
    Returns:
        Aggregated global model
    """
    global_dict = client_models[0].state_dict()
    total_size = sum(client_sizes)
    
    # Initialize with zeros
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])
    
    # Weighted average
    for model, size in zip(client_models, client_sizes):
        weight = size / total_size
        for key in global_dict.keys():
            global_dict[key] += model.state_dict()[key] * weight
    
    # Create new model with aggregated weights
    global_model = SimpleNet(list(global_dict.values())[0].shape[1] if len(list(global_dict.values())[0].shape) > 1 else 41)
    global_model.load_state_dict(global_dict)
    
    return global_model


def add_noise_to_gradients(model, epsilon):
    """
    Add Gaussian noise to gradients for differential privacy
    
    Args:
        model: Neural network model
        epsilon: Privacy budget
    """
    clip_threshold = 1.0
    noise_scale = clip_threshold / epsilon
    
    for param in model.parameters():
        if param.grad is not None:
            # Clip gradients
            grad_norm = param.grad.norm(2)
            if grad_norm > clip_threshold:
                param.grad = param.grad * (clip_threshold / grad_norm)
            
            # Add noise
            noise = torch.normal(0, noise_scale, size=param.grad.shape)
            param.grad += noise


def calculate_metrics(model, X_test, y_test):
    """
    Calculate detailed metrics
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        
    Returns:
        Dictionary with metrics
    """
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs.data, 1)
    
    predicted = predicted.numpy()
    y_test = y_test_t.numpy()
    
    # Calculate metrics manually
    accuracy = (predicted == y_test).sum() / len(y_test) * 100
    
    # For attack class (label=1)
    true_positives = ((predicted == 1) & (y_test == 1)).sum()
    false_positives = ((predicted == 1) & (y_test == 0)).sum()
    false_negatives = ((predicted == 0) & (y_test == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
