import torch as T
import torch.nn as nn
import torch.nn.functional as F

class GenericModel(nn.Module):
    
    def __init__(self, num_inputs, num_actions):
        super(GenericModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,         32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32,         32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32,         32, 3, stride=2, padding=1)
        
        self.lstm = nn.LSTMCell(1152, 512)
        
        self.actor_linear = nn.Linear(512, num_actions)
        self.critic_linear = nn.Linear(512, 1)

        self._initialize_weights()
        self.clear()

    def clear(self):
        self.hidden = T.zeros((1, 512), dtype=T.float)
        self.out = T.zeros((1, 512), dtype=T.float)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, data):
        self.hidden = self.hidden.detach()
        self.out = self.out.detach()

        data = F.relu(self.conv1(data))
        data = F.relu(self.conv2(data))
        data = F.relu(self.conv3(data))
        data = F.relu(self.conv4(data))
        
        data = data.view(data.size(0), -1)
        
        self.out, self.hidden = self.lstm(data, (self.out, self.hidden))
        
        return self.actor_linear(self.out), self.critic_linear(self.out)