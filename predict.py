import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# LOAD TRAINED MODEL

model = tf.keras.models.load_model("eye_disease_model.h5")

# IMAGE PATH

img_path = "cnv_test_1002.jpg"

# LOAD IMAGE

img = image.load_img(img_path, target_size=(224, 224))

# CONVERT IMAGE TO ARRAY

img_array = image.img_to_array(img)

# NORMALIZE IMAGE

img_array = img_array / 255.0

# EXPAND DIMENSIONS

img_array = np.expand_dims(img_array, axis=0)

# PREDICTION

prediction = model.predict(img_array)

print("\nPrediction Value:", prediction[0][0])

# CLASSIFICATION

if prediction[0][0] > 0.5:
    print("\nPrediction: NORMAL Retina")
else:
    print("\nPrediction: CNV Disease")