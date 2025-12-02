import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_FOLDER = Path("keypoints_dataset")
ACTION_LABELS = ["stand", "right punch", "left punch", "block"]

WINDOW = 10      # number of frames per training example
STRIDE = 3       # how many frames to slide each step

def prepare(): 
    all_sequences = []
    all_labels = []

    for file_path in DATA_FOLDER.glob("*.npz"):
        data = np.load(file_path, allow_pickle=True)
        keypoints = data["keypoints"]      # shape: (num_frames, num_keyponts (17) , (y, x, confidence) 3) 
        label = data["label"].item()          
        
        num_frames = len(keypoints)

        if num_frames < WINDOW:
            print(file_path, " skipped; too short:", num_frames)
            continue

        # sliding window extraction
        for start in range(0, num_frames - WINDOW + 1, STRIDE):
            window = keypoints[start:start + WINDOW].copy()
            all_sequences.append(window)
            all_labels.append(ACTION_LABELS.index(label))

    X = np.array(all_sequences, dtype=np.float32)  # shape: (num_sequences, WINDOW, 17, 3)
    y = np.array(all_labels, dtype=np.int32)      # shape: (num_sequences,)
    
    return X, y


def main():
    print("Loading data...")
    
    X, y = prepare()

    print(f"\nLoaded {len(X)} sequences")
    print(f"Shape: {X.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    np.savez_compressed("model_data/train_data.npz", X=X_train, y=y_train)
    np.savez_compressed("model_data/val_data.npz", X=X_val, y=y_val)
    np.savez_compressed("model_data/test_data.npz", X=X_test, y=y_test)
    np.savez_compressed("model_data/metadata.npz", labels=ACTION_LABELS) 

    print("\n✓ Preprocessing complete!")
    print(f"Data saved to: model_data/")


if __name__ == "__main__":
    main() 
