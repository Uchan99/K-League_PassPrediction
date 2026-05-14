# src/models_multimodal_v8_team.py
"""
MultiModalNetV8Team: Team Only (No Position Zone)

Based on MultiModalNetV6 from Tuned H96
Added: team_emb (dim=8) only
Total: 3 embeddings (type=8, result=8, team=8)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.sa = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        x = self.features(x)
        sa_map = self.sa(x)
        x = x * sa_map
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc(x))
        return x

class LSTMAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, rnn_output):
        attn_weights = torch.softmax(self.attention(rnn_output), dim=1)
        return torch.sum(attn_weights * rnn_output, dim=1)

class MultiModalNetV8Team(nn.Module):
    """
    V8-Team: Team Only (No Position Zone from coordinates)
    
    Args:
        input_dim_cont: Continuous feature dimension (10)
        num_types: Number of type categories
        num_results: Number of result categories
        num_teams: Number of team categories
        gru_hidden: GRU hidden dimension (default: 96)
    """
    def __init__(self, input_dim_cont, num_types, num_results, 
                 num_teams, gru_hidden=96):
        super(MultiModalNetV8Team, self).__init__()
        self.cnn = ImprovedCNN()
        
        # Embeddings (3 total: type, result, team)
        self.type_emb = nn.Embedding(num_types, 8)
        self.result_emb = nn.Embedding(num_results, 8)
        self.team_emb = nn.Embedding(num_teams, 8)
        
        # Total input: cont + embeddings (8+8+8 = 24)
        total_input_dim = input_dim_cont + 8 + 8 + 8
        self.gru_hidden = gru_hidden
        
        # Split GRU (Anti-Leakage) - Same as Tuned H96
        self.gru_fwd = nn.GRU(
            input_size=total_input_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        
        self.gru_bwd = nn.GRU(
            input_size=total_input_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        
        self.gru_attn = LSTMAttention(gru_hidden * 2)
        self.gru_fc = nn.Linear(gru_hidden * 2, 128)
        
        # Deep Supervision Head
        self.aux_fc = nn.Linear(gru_hidden, 2) 
        
        self.fusion_fc = nn.Sequential(
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, cont, cat, lengths):
        """
        Args:
            img: [B, 2, H, W]
            cont: [B, L, input_dim_cont]
            cat: [B, L, 3] - [type_idx, result_idx, team_idx]
            lengths: [B]
        """
        # 1. Feature Extraction
        img_feat = self.cnn(img)
        
        # Extract all categorical features
        emb_type = self.type_emb(cat[:, :, 0])
        emb_result = self.result_emb(cat[:, :, 1])
        emb_team = self.team_emb(cat[:, :, 2])
        
        # Concatenate all features
        x_seq = torch.cat([cont, emb_type, emb_result, emb_team], dim=2)
        
        # 2. Forward GRU
        packed_fwd = pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_fwd_packed, _ = self.gru_fwd(packed_fwd)
        out_fwd, _ = pad_packed_sequence(out_fwd_packed, batch_first=True)
        
        # 3. Backward GRU (Manual Flip)
        x_seq_bwd = x_seq.clone()
        for i, length in enumerate(lengths):
            x_seq_bwd[i, :length] = x_seq[i, :length].flip(0)
            
        packed_bwd = pack_padded_sequence(x_seq_bwd, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_bwd_packed, _ = self.gru_bwd(packed_bwd)
        out_bwd, _ = pad_packed_sequence(out_bwd_packed, batch_first=True)
        
        # Reverse output back to original order
        for i, length in enumerate(lengths):
            out_bwd[i, :length] = out_bwd[i, :length].flip(0)
            
        # 4. Concatenate: [Forward, Backward]
        gru_out_combined = torch.cat([out_fwd, out_bwd], dim=2) # [B, L, H*2]
        
        # 5. Main Output
        gru_ctx = self.gru_attn(gru_out_combined)
        seq_feat = F.relu(self.gru_fc(gru_ctx))
        concat_feat = torch.cat([img_feat, seq_feat], dim=1)
        final_out = self.fusion_fc(concat_feat)
        
        # 6. Aux Output (Deep Supervision)
        aux_out = self.aux_fc(out_fwd) # [B, L, 2]
        
        return final_out, aux_out

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
    def forward(self, pred, target):
        pred_real = pred * torch.tensor([105.0, 68.0], device=pred.device)
        target_real = target * torch.tensor([105.0, 68.0], device=target.device)
        return torch.mean(torch.sqrt(torch.sum((pred_real - target_real)**2, dim=1) + 1e-6))
        
class MaskedSeqEuclideanLoss(nn.Module):
    def __init__(self):
        super(MaskedSeqEuclideanLoss, self).__init__()
        
    def forward(self, pred, target, lengths):
        # pred: [B, L, 2]
        # target: [B, L, 2]
        # lengths: [B]
        
        mask = torch.arange(pred.size(1), device=pred.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1) # [B, L, 1]
        
        pred_real = pred * torch.tensor([105.0, 68.0], device=pred.device)
        target_real = target * torch.tensor([105.0, 68.0], device=target.device)
        
        diff = pred_real - target_real
        dist = torch.sqrt(torch.sum(diff**2, dim=2) + 1e-6) # [B, L]
        
        # Masking
        dist = dist * mask.squeeze(-1)
        
        return dist.sum() / mask.sum()
