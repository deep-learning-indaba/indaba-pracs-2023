# Logistic Regression from scratch
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(size, 1))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.theta)+self.bias).flatten()
    
    def predict(self, x):
        return self.forward(x) >= 0.5
    
@torch.no_grad()  # This is to let Pytorch modify our parameters
def gradient_descent(model, learning_rate):
    model.theta -= learning_rate * model.theta.grad
    model.bias -= learning_rate * model.bias.grad

def train(model, loss_function, x, y, epochs, learning_rate, print_freq=500, verbose=True, **kwargs):
    for i in range(epochs):
        model.zero_grad()  # sets all the gradients back to 0
    
        y_pred = model(x)
        loss = loss_function(y_pred, y, **kwargs)
        if (i == 0 or (i+1) % print_freq == 0)  and verbose:
            print(f'Epoch {i+1}: loss {loss.item()}')
            
        loss.backward()  # calculates the gradient of all the parameters
        gradient_descent(model, learning_rate)