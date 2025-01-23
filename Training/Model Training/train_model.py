from model import LinearNeuralNetwork as model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_data(file):
    data = pd.read_csv(file, header=None).to_numpy()
    X = data[:, 0:-1]
    Y = data[:, -1]
    

def train_model():
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
load_data('Training/train_data.csv')