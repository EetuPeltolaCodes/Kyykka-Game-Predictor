from model import LinearNeuralNetwork
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(file):
    data = pd.read_csv(file, header=None).to_numpy()
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y

def test_model(model):
    test_X, test_Y = load_data('Predictor/test_data.csv')
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    test_Y_tensor = torch.tensor(test_Y, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_X_tensor, test_Y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.load_state_dict(torch.load('Predictor\model.pth', weights_only=False))
    model.eval() 
    correct = 0
    total = 0

    cm = np.zeros((3, 3), dtype=int)

    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        cm = confusion_matrix(test_Y, model(test_X_tensor.float()).argmax(dim=1).numpy())
        cm_display = ConfusionMatrixDisplay(cm, display_labels=np.arange(3)).plot(cmap='Blues')
        plt.show()
        ConfusionMatrixDisplay(cm, display_labels=np.arange(3)).plot(cmap='Blues')

model = LinearNeuralNetwork()
test_model(model)