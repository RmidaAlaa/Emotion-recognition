import os
import tempfile
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Build the model path relative to this script's location so it works
# on any machine regardless of where the project is cloned.
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")


class Predictor:
    def __init__(self):
        # FIXED (issue #4): load_model moved inside __init__ so a bad path
        # only fails when Predictor() is constructed, not at import time.
        self.model = load_model(_MODEL_PATH, compile=True)

    def resize(self, src):
        """Resizes the image to the size of the images that the model trained on."""
        large_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(large_img, (48, 48))

        # FIXED (issue #1): use the system temp directory instead of a
        # hardcoded developer-machine path.
        temp_path = os.path.join(tempfile.gettempdir(), "new_ml_image.jpg")
        cv2.imwrite(temp_path, resize_img)

        img = Image.open(temp_path, "r")
        pixels = np.array(list(img.getdata()))
        pixels = pixels.flatten()

        image = pixels.reshape((48, 48, 1)).astype("float32")
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, src):
        mapper = {
            0: "happy",
            1: "sad",
            2: "neutral",
        }
        # FIXED (issue #3): predict_classes() was removed in TensorFlow 2.6.
        # Use model.predict() with argmax instead.
        prediction = np.argmax(self.model.predict(src), axis=-1)[0]

        return prediction
