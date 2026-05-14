import torch
import torch.nn as nn

class PolarLoss(nn.Module):
    def __init__(self, w_distance=1.0, w_direction=2.0, w_euclidean=1.0, huber_delta=1.0):
        super(PolarLoss, self).__init__()
        self.w_distance = w_distance
        self.w_direction = w_direction
        self.w_euclidean = w_euclidean
        self.huber = nn.HuberLoss(delta=huber_delta)
    
    def forward(self, pred_d, pred_sin, pred_cos, target_polar):
        """
        Args:
            pred_d: (B,) predicted distance
            pred_sin: (B,) predicted sin(theta)
            pred_cos: (B,) predicted cos(theta)
            target_polar: (B, 3) [d, sin_theta, cos_theta]
            
        Note: We need to reconstruct Cartesian to calculate Euclidean Loss accurately
              So we'll assume standard normalization for this loss component
        """
        target_d = target_polar[:, 0]
        target_sin = target_polar[:, 1]
        target_cos = target_polar[:, 2]
        
        # 1. Coordinate-wise Polar Loss
        # Distance (Huber)
        loss_distance = self.huber(pred_d, target_d)
        
        # Direction (Cosine) with Normalization
        # Normalize vectors to ensure we only measure angular difference, not magnitude
        pred_vec_norm = torch.sqrt(pred_sin**2 + pred_cos**2 + 1e-8)
        pred_sin_n = pred_sin / pred_vec_norm
        pred_cos_n = pred_cos / pred_vec_norm
        
        # cos(θ_pred - θ_true) = cos_pred*cos_true + sin_pred*sin_true
        cos_diff = pred_cos_n * target_cos + pred_sin_n * target_sin
        loss_direction = 1 - cos_diff.mean()
        
        # 2. Euclidean Distance Loss (Distance-Aware)
        # Reconstruct normalized vectors (ignoring start_x/y as we are in relative space)
        
        # Target vectors (Target is assumed to be already scaled by d_norm)
        true_rel_x = target_d * target_cos
        true_rel_y = target_d * target_sin
        
        # Pred vectors
        pred_rel_x = pred_d * pred_cos_n
        pred_rel_y = pred_d * pred_sin_n
        
        # Euclidean Distance in Normalized Space
        # This forces the model to minimize the actual spatial distance
        euclidean_dist = torch.sqrt((pred_rel_x - true_rel_x)**2 + (pred_rel_y - true_rel_y)**2 + 1e-8)
        loss_euclidean = euclidean_dist.mean()
        
        # Total Loss
        total_loss = (self.w_distance * loss_distance + 
                      self.w_direction * loss_direction + 
                      self.w_euclidean * loss_euclidean)
        
        return total_loss
