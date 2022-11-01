import torch
from torch import nn

class SliceAttention(nn.Module):
    def __init__(self, in_channels, att_dim):
        super(SliceAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, att_dim)
        self.fc2 = nn.Linear(in_channels, att_dim)
        self.fc_score = nn.Linear(att_dim, 1)
        self.fc_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//8),
            nn.ReLU(),
            nn.Linear(in_channels//8, 1),
            nn.ReLU()
        )
    
    def forward(self, x, idx):
        # x: (batch_size, seq_len, in_channels)
        # idx: start index of images correspond to each slice in the batch (batch_size)
        # output: (len(lens), in_channels)
        # Implement additive slice attention
        bs, in_channels = x.size()
        x_v = torch.tanh(self.fc1(x))
        x_u = torch.sigmoid(self.fc2(x))
        x_score = self.fc_score(x_v * x_u).squeeze()
        
        # Masking
        slice_x = []
        start_idx = 0
        for i in range(len(idx)):
            end_idx = idx[i]
            cur_score = torch.softmax(x_score[start_idx:end_idx], dim=0).unsqueeze(1)
            cur_vec = torch.sum(x[start_idx:end_idx] * cur_score, dim=0)
            slice_x.append(cur_vec)
            start_idx = end_idx
        slice_x = torch.stack(slice_x).to(x.device)
        pred = self.fc_mlp(slice_x)
        return pred