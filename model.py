import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassSimpleClassification(nn.Module):
        def __init__(self, input_dim, output_dim, l1=512, l2=128, l3=64):
            super(MulticlassSimpleClassification, self).__init__()
            self.name = 'MulticlassSimpleClassification'
            self.layer1 = nn.Linear(input_dim, l1)
            self.layer2 = nn.Linear(l1, l2)
            self.layer3 = nn.Linear(l2, l3)
            self.out = nn.Linear(l3, output_dim)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            x = torch.sigmoid(self.layer3(x))
            x = F.softmax(self.out(x), dim=1)
            return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        # torch.nn.init.zeros_(self.layer_1.bias)


