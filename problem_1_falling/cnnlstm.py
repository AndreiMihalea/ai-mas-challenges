import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable

class SqueezenetLSTM(nn.Module):
    def __init__(self, n_layers=2, n_hidden=16, n_output=3):
        super(SqueezenetLSTM, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        inception = models.squeezenet1_1()
        inception.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(*list(inception.children())[:-1])

        # No more squeezenet
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
            
        self.lstm = nn.LSTM(
            input_size = 16928,
            hidden_size = self.n_hidden,
            num_layers = self.n_layers,
            batch_first = True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, n_output)
        )

    def forward(self, x, frames):
        batch_size, timesteps = x.size()[0], x.size()[2]
        weights = self.init_hidden(frames)

        convs = []
        for t in range(timesteps):
            conv = self.conv(x[:, :, t, :, :])
            conv = conv.view(batch_size, -1)
            convs.append(conv)
        convs = torch.stack(convs, 0)
        lstm, _ = self.lstm(convs, weights)
        logit = self.fc(lstm[-1])
        return logit
    
    def init_hidden(self, batch_size):
        hidden_a = torch.randn(self.n_layers, batch_size, self.n_hidden)
        hidden_b = torch.randn(self.n_layers, batch_size, self.n_hidden)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
