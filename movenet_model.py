import os
import tensorflow as tf
import tensorflow_hub as hub
import urllib.request

class MoveNetModel:
    def __init__(self, model_name, type="tfhub") : 
        self.model_name = model_name
        self.type = type
       
        if(self.type == "tfhub"):
            self._load_tfhub_model()
        elif(self.type == "tflite"):
            self._load_tflite_model()
        else:
            raise ValueError(f"Unsupported type: {self.type}")


    def _load_tfhub_model(self):
        """Load a TF Hub MoveNet model."""
        if "movenet_lightning" in self.model_name:
            self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
        elif "movenet_thunder" in self.model_name:
            self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def _load_tflite_model(self):
        """Download and load a TFLite MoveNet model."""
        if "movenet_lightning_f16" in self.model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"
            self.input_size = 192
        elif "movenet_thunder_f16" in self.model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            self.input_size = 256
        elif "movenet_lightning_int8" in self.model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
            self.input_size = 192
        elif "movenet_thunder_int8" in self.model_name:
            url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"
            self.input_size = 256
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        if not os.path.exists("model.tflite"):
            urllib.request.urlretrieve(url, "model.tflite")

        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()
    
    def predict(self, input_image):
        """
        Runs MoveNet inference on a single image.

        Args:
            input_image: A [1, height, width, 3] tensor.
        
        Returns:
            A [1, 1, 17, 3] numpy array of keypoint coordinates and scores.
        """
        if(self.type == "tflite"):
            input_image = tf.cast(input_image, tf.uint8)
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
            self.interpreter.invoke()
            keypoints = self.interpreter.get_tensor(output_details[0]['index'])
        else:
            model = self.module.signatures['serving_default']
            input_image = tf.cast(input_image, tf.int32)
            outputs = model(input_image)
            keypoints = outputs['output_0'].numpy()

        return keypoints
  

