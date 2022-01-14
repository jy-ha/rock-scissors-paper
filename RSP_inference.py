import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

TEST_IMAGE = "./test/test-1.jpg"
LABELS = np.array(['rock', 'scissors', 'paper'])

img = tf.io.read_file(TEST_IMAGE)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [64, 64])
img = tf.cast(img, tf.float32)
img = img / 255
img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)

loaded = tf.saved_model.load("./save/model_final")
infer = loaded.signatures["serving_default"]
predict = infer(tf.constant(img))['output_1']

print(predict)
decoded = LABELS[np.argmax(predict)]

print(decoded)

