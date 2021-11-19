import tensorflow as tf
import keras.metrics
from keras.models import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

from keras.applications import *

def get_compiled_model(image_height, image_width, print_summary=False):
    pretrained_model = tf.keras.applications.InceptionV3(input_shape=[image_height, image_width, 3], include_top=False)
    pretrained_model.trainable = False

    model = Sequential([
        pretrained_model,
        Flatten(),
        Dense(1024, activation="relu", kernel_regularizer="l1"),
        Dense(10, activation='softmax')
    ])

    if print_summary:
        model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(),
        metrics=[keras.metrics.Accuracy(), keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model

def train_model(compiled_model, train_imgs, train_df, validation_imgs, validation_df):
    return compiled_model.fit(train_imgs, train_df, batch_size=32, validation_data = (validation_imgs, validation_df),
            epochs=20, verbose=1)
