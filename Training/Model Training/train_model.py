from model import LinearNeuralNetwork as model
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

def load_data(file):
    with open(file, 'r', newline='') as f:
        data = pd.read_csv(f, header=None).to_numpy()
        X = data[:, 0:-1]
        Y = data[:, -1]
        print(X)
        print(Y)
    

def train_model():
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
load_data('Training\train_data.csv')