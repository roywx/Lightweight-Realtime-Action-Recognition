import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

class KeypointDataset(Dataset):
    def __init__(self, npz_path, augment):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]          # shape: (N, T, 17, 3)
        self.y = torch.tensor(data["y"], dtype=torch.long)          # shape: (N,)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        
        if(self.augment):
            x = augment(x)
        
        x = normalize_skeleton_sequence(x)
        x = x.reshape(x.shape[0], -1)
        x = add_velocity(x)

        return torch.tensor(x, dtype=torch.float32), self.y[idx]

def create_dataloaders(data_dir="model_data", batch_size=8):
    data_dir = Path(data_dir)

    tr_dataset = KeypointDataset(data_dir / "train_data.npz", augment=True)
    val_dataset   = KeypointDataset(data_dir / "val_data.npz", augment=False)
    te_dataset  = KeypointDataset(data_dir / "test_data.npz", augment=False)

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

    return tr_loader, val_loader, te_loader

def load_metadata(meta_path: Path):
    meta = np.load(meta_path, allow_pickle=True)
    labels = meta["labels"].tolist()     
    return labels

def normalize_skeleton_sequence(keypoints):
    """
    keypoints: (seq_len, 17, 3) - y, x, confidence
    """
    normalized = keypoints.copy()
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    
    for t in range(len(keypoints)):
        frame = normalized[t]
        
        # Center on hip midpoint
        hip_center = (frame[LEFT_HIP, :2] + frame[RIGHT_HIP, :2]) / 2
        frame[:, :2] -= hip_center
        
        # Scale by torso length
        shoulder_center = (frame[LEFT_SHOULDER, :2] + frame[RIGHT_SHOULDER, :2]) / 2
        torso_length = np.linalg.norm(shoulder_center)
        if torso_length > 1e-4:
            frame[:, :2] /= torso_length
    
    return normalized

def add_velocity(sequence):
    """
    sequence: (seq_len, 51) flattened
    returns: (seq_len, 85)
    """
    T = sequence.shape[0]

    # Reshape to (T, 17, 3)
    seq_reshaped = sequence.reshape(T, 17, 3)

    # Extract only (y, x) — ignore confidence
    pos = seq_reshaped[:, :, :2]          # (T, 17, 2)

    # Flatten positions to (T, 34)
    pos_flat = pos.reshape(T, -1)

    # Compute velocity
    velocity = np.zeros_like(pos_flat)
    velocity[1:] = pos_flat[1:] - pos_flat[:-1]

    # Concatenate: original 51 features + velocity (34)
    return np.concatenate([sequence, velocity], axis=-1)

def augment(keypoints):
    """keypoints: (seq_len, 17, 3)"""
    aug = keypoints.copy()
    
    # Horizontal flip (50%)
    if random.random() > 0.5:
        aug[:, :, 1] = 1.0 - aug[:, :, 1]  # flip x
        swap = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
        for l, r in swap:
            aug[:, [l, r], :] = aug[:, [r, l], :]
    
    # Scale jitter
    aug[:, :, :2] *= random.uniform(0.9, 1.1)
    
    # Joint noise
    aug[:, :, :2] += np.random.normal(0, 0.01, aug[:, :, :2].shape)
    
    return aug
