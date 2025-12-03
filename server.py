import cv2
import numpy as np
import torch
from pathlib import Path
import tensorflow as tf
import socket
import json
import time
import numpy as np

from keypoint_dataset import load_metadata, normalize_skeleton_sequence, add_velocity
from movenet_model import MoveNetModel
from gru import GRUActionClassifier
from collections import deque, Counter
from train_generic import restore_checkpoint
from visualization_utils import draw_prediction_on_image

SEQ_LEN = 5
ACTION_LABELS = ["stand", "right punch", "left punch", "block"]
CHECKPOINT_PATH = "./checkpoints/GRU/"
METADATA_PATH = "model_data/metadata.npz"
CONFIDENCE_THRESHOLD = 0.3  

def z_score_normalize(sequence, mean, std):
    sequence_norm = sequence.copy()

    # mean, std shape is (1, 1, 17, 2)
    # add batch dimension: (1, seq_len, 17, 3)
    sequence_norm = sequence_norm[np.newaxis, ...]
    
    # Normalize (y, x) coordinates only
    sequence_norm[..., :2] = (sequence_norm[..., :2] - mean) / std
    
    # Remove batch dimension
    return sequence_norm[0]

def load_model_and_metadata():
    # Load metadata (mean, std for normalization)
    meta = np.load(METADATA_PATH, allow_pickle=True)

    labels = meta["labels"].tolist()
    
    # Load model
    model = GRUActionClassifier()
    model, start_epoch, stats = restore_checkpoint(model, CHECKPOINT_PATH)
    
    return model, labels

def predict_action(model, sequence):
    # training pipeline
    seq = normalize_skeleton_sequence(sequence)
    seq = seq.reshape(SEQ_LEN, -1)
    seq = add_velocity(seq)

    X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 20, 102)
    
    # Predict
    with torch.no_grad():
        output = model(X)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    return pred_idx, confidence, all_probs

def main():
    # initialize movenet model
    movenet = MoveNetModel(model_name="movenet_lightning_f16", type="tflite")
    # load model + meta data
    model, labels = load_model_and_metadata()
    # open webcam
    cap = cv2.VideoCapture(0)
    # rolling buffer
    buffer = deque(maxlen=SEQ_LEN) 

    current_prediction = "..."
    current_confidence = 0.0
    all_probs = np.zeros(len(ACTION_LABELS))  
    pred_idx = 0

     # set up socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 5000))
    server.listen(1)

    print("Waiting for connection")
    conn, addr = server.accept()
    print("Connected")
    last_sent_pred = None
    try:
        while(True):
            ret, frame = cap.read()
            if not ret:
                continue
            
            input_image = tf.image.resize_with_pad(frame, movenet.input_size, movenet.input_size)
            input_image = tf.expand_dims(input_image, axis=0)
            keypoints_with_scores = movenet.predict(input_image)
            keypoints = keypoints_with_scores[0, 0, :, :]
            buffer.append(keypoints)
            
            # if buffer full, run GRU
            if(len(buffer) == SEQ_LEN) :
                sequence = np.array(buffer)
                # Predict
                pred_idx, confidence, all_probs = predict_action(model, sequence)

                current_prediction = labels[pred_idx]
                current_confidence = confidence

                left_wrist = keypoints[9]   # [y, x, confidence]
                right_wrist = keypoints[10] # [y, x, confidence]

                 # if prediction != last_pred && confidence is high enough, send client message
              #  if(last_sent_pred != current_prediction and confidence >= .95) :
                message = {
                    "action": current_prediction,
                    "confidence": round(confidence, 3),
                    "left_hand": {
                        "x": round(float(left_wrist[1]), 4),  # x is index 1
                        "y": round(float(left_wrist[0]), 4),  # y is index 0
                        "confidence": round(float(left_wrist[2]), 4)
                    },
                    "right_hand": {
                        "x": round(float(right_wrist[1]), 4),
                        "y": round(float(right_wrist[0]), 4),
                        "confidence": round(float(right_wrist[2]), 4)
                    }
                    
                }
                last_sent_pred = current_prediction
                data = json.dumps(message) + "\n"
                conn.send(data.encode('utf-8'))

            # Add prediction text
            buffer_status = f"{len(buffer)}/{SEQ_LEN}"
            
            cv2.putText(
                frame, 
                f"Action: {current_prediction}", 
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 255, 0) if current_confidence > 0.7 else (0, 165, 255), 
                3
            )
            
            cv2.putText(
                frame, 
                f"Confidence: {current_confidence:.2%}", 
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            
            cv2.putText(
                frame, 
                f"Buffer: {buffer_status}", 
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (200, 200, 200), 
                2
            )

                # Show all class probabilities if buffer is full
            if len(buffer) == SEQ_LEN and current_prediction not in ["Warming up...", "Low confidence"]:
                y_pos = 180
                for i, label in enumerate(labels):
                    prob = all_probs[i]
                    bar_width = int(prob * 200)
                    color = (0, 255, 0) if i == pred_idx else (100, 100, 100)
                    
                    cv2.rectangle(
                        frame, 
                        (20, y_pos), 
                        (20 + bar_width, y_pos + 20), 
                        color, 
                        -1
                    )
                    
                    cv2.putText(
                        frame, 
                        f"{label}: {prob:.1%}", 
                        (230, y_pos + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        1
                    )
                    
                    y_pos += 30
            cv2.imshow("Action Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (BrokenPipeError, ConnectionResetError):
        print("Disconnected")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        server.close()

if __name__ == "__main__":
    main()
