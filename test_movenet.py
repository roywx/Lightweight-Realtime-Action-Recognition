import cv2
import numpy as np
import tensorflow as tf

from visualization_utils import draw_prediction_on_image
from movenet_model import MoveNetModel

# movenet_model = MoveNetModel(model_name="movenet_lightning")
movenet_model = MoveNetModel(model_name="movenet_lightning_f16", type="tflite")
cap = cv2.VideoCapture(0)


while True:
    success, frame = cap.read()
    if not success:
        break
    input_image = tf.image.resize_with_pad(frame, movenet_model.input_size, movenet_model.input_size)
    input_image = tf.expand_dims(input_image, axis=0)
    

    keypoints_with_scores = movenet_model.predict(input_image)
   
    display_image = tf.cast(tf.image.resize_with_pad(frame, 640, 360), dtype=tf.int32)

    output_overlay = draw_prediction_on_image(display_image.numpy(), keypoints_with_scores)
    
    cv2.imshow('Webcam', output_overlay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
