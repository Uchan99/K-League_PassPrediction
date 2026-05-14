"""
Polar Coordinate Inference Helper Functions
"""
import numpy as np
import torch

def normalize_vec(sin_theta, cos_theta, epsilon=1e-8):
    """
    Normalize (sin, cos) to unit vector
    
    Args:
        sin_theta: predicted sin component (tensor or numpy)
        cos_theta: predicted cos component (tensor or numpy)
        epsilon: small value to avoid division by zero
    
    Returns:
        (sin_norm, cos_norm): normalized direction components
    """
    if isinstance(sin_theta, torch.Tensor):
        norm = torch.sqrt(sin_theta**2 + cos_theta**2) + epsilon
        return sin_theta / norm, cos_theta / norm
    else:
        norm = np.sqrt(sin_theta**2 + cos_theta**2) + epsilon
        return sin_theta / norm, cos_theta / norm



# Alias for backward compatibility
normalize_direction = normalize_vec

def polar_to_cartesian(d_norm, sin_theta, cos_theta, start_x, start_y, d_scale=105.0, 
                       normalize_direction=True, clip_field=True):
    """
    Convert polar prediction to cartesian coordinates
    
    Args:
        d_norm: normalized distance prediction
        sin_theta: sin component of direction
        cos_theta: cos component of direction
        start_x: pass start x coordinate
        start_y: pass start y coordinate
        d_scale: denormalization scale (default: FIELD_X = 105.0)
        normalize_direction: whether to normalize sin/cos to unit vector
        clip_field: whether to clip to field boundaries
    
    Returns:
        (end_x, end_y): predicted end coordinates
    """
    # Normalize direction to unit vector
    if normalize_direction:
        sin_theta, cos_theta = normalize_vec(sin_theta, cos_theta)
    
    # Denormalize distance
    if isinstance(d_norm, torch.Tensor):
        d = d_norm * d_scale
        theta = torch.atan2(sin_theta, cos_theta)
        end_x = start_x + d * torch.cos(theta)
        end_y = start_y + d * torch.sin(theta)
        
        if clip_field:
            end_x = torch.clamp(end_x, 0, 105)
            end_y = torch.clamp(end_y, 0, 68)
    else:
        d = d_norm * d_scale
        theta = np.arctan2(sin_theta, cos_theta)
        end_x = start_x + d * np.cos(theta)
        end_y = start_y + d * np.sin(theta)
        
        if clip_field:
            end_x = np.clip(end_x, 0, 105)
            end_y = np.clip(end_y, 0, 68)
    
    return end_x, end_y
