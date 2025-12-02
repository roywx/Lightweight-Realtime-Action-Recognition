import cv2
import numpy as np
import tensorflow as tf
from movenet_model import MoveNetModel
from pathlib import Path
import time
from visualization_utils import draw_prediction_on_image


#ACTION_LABELS = ["stand", "walk", "jump", "punch", "block"]
ACTION_LABELS = ["right punch", "right punch", "right punch", "right punch", "right punch", "left punch", "left punch", "left punch", "left punch", "left punch"]
#ACTION_LABELS = ["right punch", "right punch", "right punch", "right punch", "right punch", "left punch", "left punch", "left punch", "left punch", "left punch", "block", "block", "block", "block", "block"]

COUNTDOWN_SEC = 5  
RECORD_SEC = 1.5    
VISUALIZATION_ON = True
MOVENET_MODEL = MoveNetModel(model_name="movenet_lightning_f16", type="tflite")

cap = cv2.VideoCapture(0)
output_folder = Path(f"keypoints_dataset")
output_folder.mkdir(parents=True, exist_ok=True)

def countdown(label):
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start_time
        remaining = COUNTDOWN_SEC - int(elapsed)
        if remaining <= 0:
            break
        
        display_image = tf.cast(tf.image.resize_with_pad(frame, 640, 360), dtype=tf.int32)
        display_image = np.ascontiguousarray(display_image.numpy(), dtype=np.uint8)
            

        cv2.putText(display_image, f"Next: {label}", (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(display_image, f"Starting in {remaining}...", (30, 120),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Recording", display_image)
        cv2.waitKey(1)
      
def record_action(label):
    print(f"--- Recording '{label}' ---")

    data = []
    start_time = time.time()
    captured = 0

    while time.time() - start_time < RECORD_SEC:
        ret, frame = cap.read()
        if not ret:
            continue

        input_image = tf.image.resize_with_pad(frame, MOVENET_MODEL.input_size, MOVENET_MODEL.input_size)
        input_image = tf.expand_dims(input_image, axis=0)

        keypoints_with_scores = MOVENET_MODEL.predict(input_image)
        keypoints_save = keypoints_with_scores[0, 0, :, :]

        data.append({
            "keypoints": keypoints_save,
        })
        
        display_image = tf.cast(tf.image.resize_with_pad(frame, 640, 360), dtype=tf.int32)

        # scaling is going to be slightly off because of draw_prediction_on_image
        # could be fixed but just need to collect data 

        output_overlay = draw_prediction_on_image(display_image.numpy(), keypoints_with_scores, output_image_height=720)
        output_overlay = np.ascontiguousarray(output_overlay, dtype=np.uint8)

        cv2.putText(output_overlay, f"Action: {label}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.putText(output_overlay, f"Recording... {time.time() - start_time}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
       
        cv2.imshow("Recording", output_overlay)
        cv2.waitKey(1)

        captured += 1

    # save data
    if(len(data) > 0): 
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        keypoints = np.array([d["keypoints"] for d in data], dtype=np.float32)

        out_path = output_folder / f"{label}_{timestamp}.npz"
        np.savez_compressed(out_path, keypoints=keypoints, label=label)

        actual_fps = len(data) / RECORD_SEC
        print(f"Saved {len(data)} frames ({actual_fps:.1f} fps)")
    else: 
        print("Error saving")

def main():
    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
    # hard set pop up position
    cv2.moveWindow("Recording", 600, 50)

    for action in ACTION_LABELS : 
        countdown(action)
        record_action(action)
    
    print("completed")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 


