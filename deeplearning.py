import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing import image
import base64

BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
STEPS_PER_EPOCH = 5
train_dir = pathlib.Path('train/')
valid_dir = pathlib.Path('validation/')
CLASS_NAMES = []
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])


class DeepLearning(object):
    def get_images(self):
        # The 1./255 is to convert from uint8 to float32 in range [0,1].
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True,
            # zoom_range=0.5
        )
        image_generator2 = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True,
            # zoom_range=0.5
        )
        train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                             batch_size=BATCH_SIZE,
                                                             shuffle=True,
                                                             target_size=(
                                                                 IMG_HEIGHT, IMG_WIDTH),
                                                             classes=list(CLASS_NAMES))
        valid_data_gen = image_generator2.flow_from_directory(directory=str(valid_dir),
                                                             batch_size=BATCH_SIZE,
                                                             shuffle=True,
                                                             target_size=(
                                                                 IMG_HEIGHT, IMG_WIDTH),
                                                             classes=list(CLASS_NAMES))
        return train_data_gen, valid_data_gen

    def training_IA(self):
        train_data_gen, valid_data_gen = DeepLearning.get_images(self)
        image_train, label_train = next(train_data_gen)
        image_valid, label_valid = next(valid_data_gen)

        model = Sequential([
            Conv2D(16, (3, 3), padding='same', activation='relu',
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(2)
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()

        history = model.fit(image_train, label_train, epochs=10, steps_per_epoch=3,
                            validation_data=(image_valid, label_valid), validation_steps=3)

        model.save("model.h5")

        test_image = image.load_img("predictimg.jpeg", target_size = (128, 128)) 
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        predict = model.predict(test_image)
        print(CLASS_NAMES[np.argmax(predict[0])], predict)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(10)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        pass

    def preditc_IA(self, image_request):
        imgdata = base64.b64decode(image_request)
        with open("predictimg.jpeg", 'wb') as f:
            f.write(imgdata)
        test_image = image.load_img("predictimg.jpeg", target_size = (128, 128)) 
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        model = load_model('model.h5')
        predict = model.predict(test_image)
        return CLASS_NAMES[np.argmax(predict[0])]

ia = DeepLearning()
ia.training_IA()
#ia.preditc_IA('ssss')

