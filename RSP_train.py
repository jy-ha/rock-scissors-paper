import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Model


class RSP_Model(Model):
    def __init__(self):
        super(RSP_Model, self).__init__()
        self.maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv1 = Conv2D(64, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')
        self.conv2 = Conv2D(64, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')
        self.conv3 = Conv2D(128, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')
        self.conv4 = Conv2D(128, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.d1 = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu')
        self.d2 = Dense(3, activation='softmax')

    def call(self, x, train=True):
        x = self.conv1(x)

        x = self.maxpool(x)
        x = self.conv2(x)

        x = self.maxpool(x)
        x = self.conv3(x)

        x = self.maxpool(x)
        x = self.conv4(x)

        x = self.maxpool(x)
        x = self.flatten(x)
        if train:
            x = self.dropout(x)
        x = self.d1(x)

        return self.d2(x)


path_train = './data/train/'
path_test = './data/test/'

LABELS = np.array(['rock', 'scissors', 'paper'])
BATCH_SIZE = 32
IMAGE_SIZE = 64
EPOCHS = 30
IMAGE_TYPE = 'jpg'

def preview_dataset(dataset):
    plt.figure(figsize=(12, 12))
    plot_index = 0
    for features in dataset.take(12):
        (image, label) = features
        plot_index += 1
        plt.subplot(3, 4, plot_index)
        plt.imshow(image.numpy())
    plt.show()

def generate_examples(path, shuffle_buffer_size=1000):
    list_ds = tf.data.Dataset.list_files(path + '*/*.' + IMAGE_TYPE)

    def process_files(file_path):
        # process label
        parts = tf.strings.split(file_path, os.path.sep)
        data_Y = parts[-2] == LABELS
        # process image
        img = tf.io.read_file(file_path)
        if(IMAGE_TYPE == 'jpg'):    img = tf.image.decode_jpeg(img, channels=3)
        elif(IMAGE_TYPE == 'png'):  img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32)
        img = img / 255
        data_X = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)
        return data_X, data_Y

    dataset =  list_ds.map(process_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if isinstance(True, str):
        dataset = dataset.cache(True)
    else:
        dataset = dataset.cache()

    def image_augmentation(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image, max_delta=0.08)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1)
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        return image

    dataset = dataset.repeat(20)
    dataset = dataset.map(lambda x, y: (image_augmentation(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    preview_dataset(dataset)

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


model = RSP_Model()
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

train_ds = generate_examples(path_train)
test_ds = generate_examples(path_test)


def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(images, labels):
    predictions = model(images, False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in tqdm(range(EPOCHS)):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'epoch: {}, loss_train: {}, acc_train: {}, loss_test: {}, acc_test: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))

tf.saved_model.save(model, './save/model_final')
#model.save('./save/model_final')

model.summary()