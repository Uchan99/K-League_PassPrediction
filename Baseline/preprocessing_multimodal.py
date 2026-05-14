import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

FIELD_X = 105.0
FIELD_Y = 68.0
GRID_SIZE = 2.0  
# Continuity Threshold (meters)
CONTINUITY_THRESHOLD = 2.0

class FootballPreprocessorMultimodal:
    """
    Multimodal Preprocessor (Refactored V8 Logic)
    - Explicit dx/dy (Vector based)
    - Continuity Handling (Gap > 2m -> Mask physics)
    - Optimized for V2.5/Multimodal Architecture
    """
    def __init__(self):
        self.type_encoder = LabelEncoder()
        self.result_encoder = LabelEncoder()
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def fit(self, df):
        """
        Fit scaler and encoders on training data.
        """
        df = df.copy()
        
        # Handle Implicits if needed (Legacy support)
        # Note: If 'result_name' is missing, it creates issues.
        if 'result_name' in df.columns:
            mask = df['result_name'].isna() | (df['result_name'] == '')
            if mask.sum() > 0:
                df.loc[mask, 'result_name'] = df.loc[mask, 'type_name'].apply(lambda x: f"{x}_Implicit")

        unique_types = df['type_name'].astype(str).unique().tolist()
        unique_results = df['result_name'].astype(str).unique().tolist() if 'result_name' in df.columns else []
        
        self.type_encoder.fit(unique_types + ['Unknown'])
        self.result_encoder.fit(unique_results + ['Unknown'])
        
        # Generate features for scaler fitting
        temp_df = self._engineer_features(df)
        
        # Feature definition (V8 style but without embeddings)
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
            
        # Rounding (Safety)
        cols_to_round = [c for c in ['start_x', 'start_y', 'end_x', 'end_y'] if c in df.columns]
        if cols_to_round:
            df[cols_to_round] = df[cols_to_round].round()
            
        # Sort
        if 'time_seconds' in df.columns:
            df = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
        
        # 1. Basic Norms
        df['start_x_norm'] = df['start_x'] / FIELD_X
        df['start_y_norm'] = df['start_y'] / FIELD_Y
        
        # Dist to goal
        # Goal center is at (105, 34)
        df['dist_to_goal'] = np.sqrt((FIELD_X - df['start_x'])**2 + ((FIELD_Y/2) - df['start_y'])**2)
        df['dist_to_goal_norm'] = df['dist_to_goal'] / FIELD_X
        
        # 2. Previous Event Info & Continuity
        # We need to reconstruct 'prev_end' from shift(1)
        
        # Note: In inference, if end_x is NOT in columns, we assume we are predicting final step?
        # But 'transform' is usually called on the SEQUENCE of events leading up to target.
        # So 'end_x' usually exists for all PAST events in the sequence.
        
        if 'end_x' in df.columns:
            df['end_x_norm'] = df['end_x'] / FIELD_X
            df['end_y_norm'] = df['end_y'] / FIELD_Y
             
            df['end_x_prev'] = df.groupby('game_episode')['end_x'].shift(1).fillna(df['start_x'])
            df['end_y_prev'] = df.groupby('game_episode')['end_y'].shift(1).fillna(df['start_y'])
        else:
             # Fallback for when end_x is missing (should not happen in sequence history)
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

        # 3. Physics (prev)
        if 'end_x' in df.columns:
            df['curr_dx'] = df['end_x'] - df['start_x']
            df['curr_dy'] = df['end_y'] - df['start_y']
            df['curr_dist'] = np.sqrt(df['curr_dx']**2 + df['curr_dy']**2)
            
            # Time delta
            df['time_delta'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0.1)
            df['time_delta'] = df['time_delta'].apply(lambda x: max(x, 0.01))
            
            df['curr_speed'] = df['curr_dist'] / df['time_delta']
            
            # Shift to get Prev
            df['dx_prev'] = df.groupby('game_episode')['curr_dx'].shift(1).fillna(0)
            df['dy_prev'] = df.groupby('game_episode')['curr_dy'].shift(1).fillna(0)
            df['speed_prev'] = df.groupby('game_episode')['curr_speed'].shift(1).fillna(0)
        else:
            df['dx_prev'] = 0
            df['dy_prev'] = 0
            df['speed_prev'] = 0
            df['time_delta'] = 1.0

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
            
        # 5. Categorical Indices
        if hasattr(self, 'type_encoder'):        
            df['type_idx'] = df['type_name'].astype(str).map(
                lambda x: self.type_encoder.transform([x])[0] if x in self.type_encoder.classes_ else self.type_encoder.transform(['Unknown'])[0]
            )
            df['result_idx'] = df['result_name'].astype(str).map(
                lambda x: self.result_encoder.transform([x])[0] if x in self.result_encoder.classes_ else self.result_encoder.transform(['Unknown'])[0]
            )
        
        return df

    def transform(self, df, is_train=True):
        df = self._engineer_features(df)
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        episodes = []
        if 'game_id' in df.columns:
            grouped = df.groupby(['game_id', 'game_episode'])
        else:
            grouped = df.groupby('game_episode')
            
        for name, g in grouped:
            # Drop episodes that are too short? Masking sequence length?
            # V2.5 logic: len(g) < 2 continue for train.
            if is_train and len(g) < 2:
                continue
                
            f_cont = g[self.feature_columns].values.astype(np.float32)
            f_cat = g[['type_idx', 'result_idx']].values.astype(np.int64)
            
            if is_train:
                target = g.iloc[-1][['end_x', 'end_y']].values.astype(np.float32)
                target_norm = g.iloc[-1][['end_x_norm', 'end_y_norm']].values.astype(np.float32)
            else:
                target = np.array([0, 0], dtype=np.float32)
                target_norm = np.array([0, 0], dtype=np.float32)
            
            game_episode = g.iloc[0]['game_episode'] if 'game_episode' in g.columns else 0
            
            episodes.append({
                'cont': f_cont,
                'cat': f_cat,
                'target': target_norm,   
                'target_raw': target,    
                'game_episode': game_episode
            })
            
        return episodes

    def get_input_dim(self):
        return len(self.feature_columns)
    
    def get_num_classes(self):
        return len(self.type_encoder.classes_), len(self.result_encoder.classes_)
