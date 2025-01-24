from model import LinearNeuralNetwork
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def load_data(file):
    data = pd.read_csv(file, header=None).to_numpy()
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y
    

def train_model(num_epochs, batch_size):
    train_X, train_Y = load_data('Training/train_data.csv')
    validation_X, validation_Y = load_data('Training/validation_data.csv')
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.long)
    validation_X = torch.tensor(validation_X, dtype=torch.float32)
    validation_Y = torch.tensor(validation_Y, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
    validation_dataset = torch.utils.data.TensorDataset(validation_X, validation_Y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    model = LinearNeuralNetwork()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = np.inf
    val_losses = []
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Convert inputs and labels to float32
            inputs = inputs.float()
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item()
        scheduler.step(running_loss / len(train_loader))

        # Epoch validation loss
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            for val_inputs, val_labels in valid_loader:
                val_inputs = val_inputs.float()
                val_labels = val_labels.long()
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()
            val_running_loss /= len(valid_loader)

        val_losses.append(val_running_loss)
        train_losses.append(running_loss / len(train_loader))
        if val_running_loss < best_val_loss:
            best_val_loss = val_running_loss
            torch.save(model.state_dict(), 'Training/model.pth')
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_running_loss:.4f}')

    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    train_model(40, 32)
    
    
