from torch import nn

class LinearNeuralNetwork(nn.Module):
    def __init__(self):
        super(LinearNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc3(x))
        x = self.log_softmax(x)
        return x
             