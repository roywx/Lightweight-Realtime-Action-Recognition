import torch
import torch.nn as nn

class GRUActionClassifier(nn.Module):
    def __init__(self, input_size=85, hidden_size=128, num_layers=2, num_classes=4, dropout=0.4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True 
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.0)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param) 

    def forward(self, x):
        # x: (batch_size, seq_len, 17 * 3)

        # out: (batch, seq_len, hidden_size)
        gru_output, _ = self.gru(x)                       
        #last_timestep = output[:, -1, :]     
        attn_weights = torch.softmax(self.attention(gru_output), dim=1)
        context = (attn_weights * gru_output).sum(dim=1)

         # (batch, num_classes)
        out = self.fc(self.dropout(context))                        
        return out