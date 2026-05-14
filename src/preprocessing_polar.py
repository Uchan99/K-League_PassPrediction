import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

FIELD_X = 105.0
FIELD_Y = 68.0
GRID_SIZE = 2.0  
# Continuity Threshold (meters)
CONTINUITY_THRESHOLD = 2.0
# Distance normalization denominator
D_NORM_DENOM = FIELD_X  # Use field length for normalization

class FootballPreprocessorPolar:
    """
    Polar Coordinate Preprocessor
    - Converts (end_x, end_y) → (distance, sin_theta, cos_theta)
    - Based on preprocessing_multimodal_fixed.py
    - No data leakage: polar labels only generated in train mode
    """
    def __init__(self):
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Fast lookup dicts (populated after fit)
        self.type_map = {}
        self.result_map = {}
        
    def fit(self, df):
        """
        Fit scaler and encoders on training data.
        """
        df = df.copy()
        
        # Handle Implicits if needed (Legacy support)
        if 'result_name' in df.columns:
            mask = df['result_name'].isna() | (df['result_name'] == '')
            if mask.sum() > 0:
                df.loc[mask, 'result_name'] = df.loc[mask, 'type_name'].apply(lambda x: f"{x}_Implicit")

        unique_types = df['type_name'].astype(str).unique().tolist()
        unique_results = df['result_name'].astype(str).unique().tolist() if 'result_name' in df.columns else []
        
        self.type_encoder.fit(unique_types + ['Unknown'])
        self.result_encoder.fit(unique_results + ['Unknown'])
        
        # Build fast lookup dicts
        self.type_map = {cls: idx for idx, cls in enumerate(self.type_encoder.classes_)}
        self.result_map = {cls: idx for idx, cls in enumerate(self.result_encoder.classes_)}
        
        # Generate features for scaler fitting
        temp_df = self._engineer_features(df)
        
        # Feature definition (same as fixed preprocessor)
        self.feature_columns = [
            'start_x_norm', 'start_y_norm',      # 0, 1
            'end_x_prev_norm', 'end_y_prev_norm',# 2, 3
            'dx_prev_norm', 'dy_prev_norm',      # 4, 5
            'speed_prev_log',                    # 6
            'time_delta',                        # 7
            'dist_to_goal_norm',                 # 8
            'is_continuous'                      # 9
        ]
        
        self.scaler.fit(temp_df[self.feature_columns])
        return self

    def _engineer_features(self, df):
        df = df.copy()
        
        # Implicit handling
        if 'result_name' in df.columns:
            mask = df['result_name'].isna() | (df['result_name'] == '')
            if mask.sum() > 0:
                df.loc[mask, 'result_name'] = df.loc[mask, 'type_name'].apply(lambda x: f"{x}_Implicit")
            
        # Sort
        if 'time_seconds' in df.columns:
            df = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
        
        # 1. Basic Norms
        df['start_x_norm'] = df['start_x'] / FIELD_X
        df['start_y_norm'] = df['start_y'] / FIELD_Y
        
        # Dist to goal
        df['dist_to_goal'] = np.sqrt((FIELD_X - df['start_x'])**2 + ((FIELD_Y/2) - df['start_y'])**2)
        df['dist_to_goal_norm'] = df['dist_to_goal'] / FIELD_X
        
        # 2. Previous Event Info & Continuity
        if 'end_x' in df.columns:
            df['end_x_norm'] = df['end_x'] / FIELD_X
            df['end_y_norm'] = df['end_y'] / FIELD_Y
             
            df['end_x_prev'] = df.groupby('game_episode')['end_x'].shift(1).fillna(df['start_x'])
            df['end_y_prev'] = df.groupby('game_episode')['end_y'].shift(1).fillna(df['start_y'])
        else:
             # Fallback for when end_x is missing
             df['end_x_prev'] = df['start_x'] 
             df['end_y_prev'] = df['start_y']
        
        df['gap'] = np.sqrt(
            (df['start_x'] - df['end_x_prev'])**2 + 
            (df['start_y'] - df['end_y_prev'])**2
        )
        
        # Continuity Flag
        df['is_continuous'] = (df['gap'] < CONTINUITY_THRESHOLD).astype(float)
        
        # Normalize prev coords
        df['end_x_prev_norm'] = df['end_x_prev'] / FIELD_X
        df['end_y_prev_norm'] = df['end_y_prev'] / FIELD_Y

        # 3. Physics (prev) & Time Delta
        # Always create time_delta first for consistency
        if 'time_seconds' in df.columns:
            df['time_delta'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0.1)
            df['time_delta'] = df['time_delta'].apply(lambda x: max(x, 0.01))
        else:
            df['time_delta'] = 1.0
        
        if 'end_x' in df.columns:
            df['curr_dx'] = df['end_x'] - df['start_x']
            df['curr_dy'] = df['end_y'] - df['start_y']
            df['curr_dist'] = np.sqrt(df['curr_dx']**2 + df['curr_dy']**2)
            
            df['curr_speed'] = df['curr_dist'] / df['time_delta']
            
            # Shift to get Prev
            df['dx_prev'] = df.groupby('game_episode')['curr_dx'].shift(1).fillna(0)
            df['dy_prev'] = df.groupby('game_episode')['curr_dy'].shift(1).fillna(0)
            df['speed_prev'] = df.groupby('game_episode')['curr_speed'].shift(1).fillna(0)
        else:
            df['dx_prev'] = 0
            df['dy_prev'] = 0
            df['speed_prev'] = 0

        # Log Speed
        df['speed_prev_log'] = np.log1p(df['speed_prev'])
        
        # Norm DX/DY
        df['dx_prev_norm'] = df['dx_prev'] / FIELD_X
        df['dy_prev_norm'] = df['dy_prev'] / FIELD_Y
        
        # 4. MASKING (Anti-Leakage / Denoising)
        mask_discont = df['is_continuous'] == 0
        df.loc[mask_discont, 'speed_prev_log'] = 0
        df.loc[mask_discont, 'dx_prev_norm'] = 0
        df.loc[mask_discont, 'dy_prev_norm'] = 0
        
        # Handle Inf/Nan
        features = ['speed_prev_log', 'dx_prev_norm', 'dy_prev_norm', 'dist_to_goal_norm']
        for col in features:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
            
        # 5. Categorical Indices (Fast dict lookup)
        if hasattr(self, 'type_map'):
            unknown_type_idx = self.type_map.get('Unknown', 0)
            unknown_result_idx = self.result_map.get('Unknown', 0)
            
            df['type_idx'] = df['type_name'].astype(str).map(self.type_map).fillna(unknown_type_idx).astype(int)
            df['result_idx'] = df['result_name'].astype(str).map(self.result_map).fillna(unknown_result_idx).astype(int)
        
        return df

    def _create_polar_target(self, row):
        """
        Convert (start_x, start_y, end_x, end_y) → (d_norm, sin_theta, cos_theta)
        
        Returns:
            np.array([d_norm, sin_theta, cos_theta])
            - d_norm: normalized distance (0 ~ ~1.2)
            - sin_theta, cos_theta: direction components
        """
        dx = row['end_x'] - row['start_x']
        dy = row['end_y'] - row['start_y']
        
        # Distance (normalized)
        d = np.sqrt(dx**2 + dy**2)
        d_norm = d / D_NORM_DENOM  # Normalize to similar scale as sin/cos
        
        # Direction
        theta = np.arctan2(dy, dx)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        return np.array([d_norm, sin_theta, cos_theta], dtype=np.float32)

    def transform(self, df, is_train=True):
        df = self._engineer_features(df)
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        episodes = []
        if 'game_id' in df.columns:
            grouped = df.groupby(['game_id', 'game_episode'])
        else:
            grouped = df.groupby('game_episode')
            
        for name, g in grouped:
            if is_train and len(g) < 2:
                continue
                
            f_cont = g[self.feature_columns].values.astype(np.float32)
            f_cat = g[['type_idx', 'result_idx']].values.astype(np.int64)
            
            # Store start coordinates (needed for inference)
            start_x = g.iloc[-1]['start_x']
            start_y = g.iloc[-1]['start_y']
            
            if is_train:
                if 'end_x' in g.columns:
                    # Cartesian target (for OOF evaluation)
                    target = g.iloc[-1][['end_x', 'end_y']].values.astype(np.float32)
                    target_norm = g.iloc[-1][['end_x_norm', 'end_y_norm']].values.astype(np.float32)
                    
                    # Polar target (for training)
                    target_polar = self._create_polar_target(g.iloc[-1])
                else:
                    target = np.array([0, 0], dtype=np.float32)
                    target_norm = np.array([0, 0], dtype=np.float32)
                    target_polar = np.array([0, 0, 1], dtype=np.float32)  # d=0, sin=0, cos=1
            else:
                target = np.array([0, 0], dtype=np.float32)
                target_norm = np.array([0, 0], dtype=np.float32)
                target_polar = np.array([0, 0, 1], dtype=np.float32)
            
            game_episode = g.iloc[0]['game_episode'] if 'game_episode' in g.columns else 0
            game_id = g.iloc[0]['game_id'] if 'game_id' in g.columns else 0
            
            episodes.append({
                'cont': f_cont,
                'cat': f_cat,
                'target': target_norm,         # For compatibility
                'target_raw': target,          # For OOF evaluation
                'target_polar': target_polar,  # For training (d_norm, sin, cos)
                'start_x': start_x,            # For inference
                'start_y': start_y,            # For inference
                'd_scale': float(D_NORM_DENOM),  # For denormalization in inference
                'game_episode': game_episode,
                'game_id': game_id             # [Fix] Added for inference merging
            })
            
        return episodes

    def get_input_dim(self):
        return len(self.feature_columns)
    
    def get_num_classes(self):
        return len(self.type_encoder.classes_), len(self.result_encoder.classes_)
