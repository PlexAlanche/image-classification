import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import numpy as np

model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)

img_path = ''
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


predictions = model.predict(img_array)

decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())

top_prediction = decoded_predictions[0][0]
class_name = top_prediction[1]
confidence = top_prediction[2]

print("Output: {} (Confidence: {:.2f})".format(class_name, confidence))
