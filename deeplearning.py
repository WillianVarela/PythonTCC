import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import base64
import itertools
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, experimental
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 32 # epocas * steps
IMG_HEIGHT = 200
IMG_WIDTH = 200
EPOCHS = 100
STEPS_TRAIN = 10
STEPS_VAL = 5
train_dir = pathlib.Path('train/')
valid_dir = pathlib.Path('validation/')
CLASS_NAMES = []
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])

class DeepLearning(object):

    def get_images(self):
        # Normalizando as imagens
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )
        image_generator2 = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )
        # Carregando as imagens do diretorio, passando o batch e tamanho assim como a classe
        train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                             batch_size=BATCH_SIZE,
                                                             target_size=(
                                                                 IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode="categorical"
                                                             )
        valid_data_gen = image_generator2.flow_from_directory(directory=str(valid_dir),
                                                             batch_size=BATCH_SIZE,
                                                             target_size=(
                                                                 IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode="categorical"
                                                             )
        return train_data_gen, valid_data_gen

    def training_IA(self):
        # DataAgumentation, flip das imagens
        data_augmentation = tf.keras.Sequential([
            experimental.preprocessing.RandomFlip("horizontal_and_vertical")
        ])

        model = Sequential([
            data_augmentation, # Realizar considerações para oque melhorou ou pirou para esta função
            Conv2D(32, (2, 2), padding='same', activation='relu',
                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 
            MaxPooling2D((2, 2)),
            Dropout(0.2), 
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(256, activation='sigmoid', kernel_initializer='he_normal'),
            Dropout(0.4),
            Dense(2, activation='sigmoid') 
        ])
        model.compile(optimizer='adam', # otimazar adam padrão
                      loss='binary_crossentropy', # cassificação binaria
                      metrics=['accuracy']) # accuracy simples
        # ModelCheckpoint - Salva o modelo com menor loss, ou seja o melhor modelo treinado
        mdlckpt = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)

        # EarlyStop -> Se o loss não diminuir para o treinamento, pois a rede não está aprendendo mais (se não diminuir em 4 epocos ele para)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, mode='min', verbose=1)

        train_data_gen, valid_data_gen = DeepLearning.get_images(self)
        image_train, label_train = next(train_data_gen)
        image_valid, label_valid = next(valid_data_gen)
        history = model.fit(image_train, label_train, epochs=EPOCHS, steps_per_epoch=STEPS_TRAIN,
                            validation_data=(image_valid, label_valid), validation_steps=STEPS_VAL, callbacks=[mdlckpt, earlystop])
        model.summary()
        model.save("model.h5")

        self.matrix_confusion(model, valid_data_gen)
        self.plot_result(history)
        pass

    def matrix_confusion(self, model, valid_data_gen):
        target_names = ['Ferrugem', 'Sadia']
        cm_title = 'Confusion Matrix'
        tick_marks = np.arange(2)
        Y_pred = model.predict(valid_data_gen, STEPS_VAL)
        y_pred = np.argmax(Y_pred, axis=1)
        cm = confusion_matrix(valid_data_gen.classes, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(cm_title)
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Label Verdadeiro')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        print(cm_title)
        print(cm)
        print('Classification Report')
        print(classification_report(valid_data_gen.classes, y_pred, target_names=target_names))
    pass

    def plot_result(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(history.epoch))

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
        imgdata = base64.b64decode(image_request) # decodifica a image de base64
        with open("predictimg.png", 'wb') as f: # salva a imagem no diretorio do projeto
            f.write(imgdata)
        # Restaura a imagem utilizando keras, normalizar esses dados
        test_image = tf.keras.preprocessing.image.load_img("predictimg.png", target_size = (200, 200))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image) # Converte a imagem para um array
        test_image = np.expand_dims(test_image, axis = 0) # expande o array
        model = load_model('model.h5') # restaura o modelo
        predict = model.predict(test_image) # realiza a predição
        print(CLASS_NAMES[np.argmax(predict[0])], predict[0])
        return '{"ferrugem": ' + str(predict[0][0]) + ', "sadia": ' + str(predict[0][1]) + '}'

ia = DeepLearning()
ia.training_IA()
# ia.preditc_IA('ssss')