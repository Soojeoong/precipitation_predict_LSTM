# LSTM.py
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers # LSTM stack layer (위로 쌓임)
        # self.input_size = input_size # feature 개수
        self.hidden_size = hidden_size # 옆으로
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights() # 가중치 초기화
        
    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # out: [batch_size, sequence_length, hidden_size]
        # Use the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier 초기화
            elif 'bias' in name:
                nn.init.zeros_(param)  # 편향을 0으로 초기화
    
if __name__ == "__main__":
    # Batch size = 32, Sequence length = 5, Input size = 3
    input_tensor = torch.randn(32, 5, 3) 
    
    # Initialize the model
    model = LSTM(input_size=3, hidden_size=4, num_layers=2, output_size=1)
    
    # Forward pass
    output = model(input_tensor)
    print(f"Final Output shape: {output.shape}")
