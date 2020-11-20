import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pathlib
import base64
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, experimental, GaussianNoise
from sklearn.metrics import classification_report, confusion_matrix
import cv2, os
from tensorflow.keras.datasets import mnist
dim_data = 0
BATCH_SIZE = 32
IMG_HEIGHT = 200
IMG_WIDTH = 200
EPOCHS = 50
train_dir = pathlib.Path('train/')
test_dir = pathlib.Path('test/')
test_dir2 = pathlib.Path('teste2/')
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])
tf.random.set_seed(1)

class DeepLearning(object):

    def get_images(self, channels=1):
        # Carregando as imagens do diretorio, passando o batch e tamanho assim como a classe
        train_data_gen = ImageDataGenerator().flow_from_directory(directory=str(train_dir), batch_size=775,
                                                             target_size=(
                                                                 IMG_HEIGHT, IMG_WIDTH),
                                                                 class_mode="categorical"
                                                             )

        teste_data_gen = ImageDataGenerator().flow_from_directory(directory=str(test_dir), batch_size=125,
                                                                  target_size=(
                                                                      IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode="categorical"
                                                                  )

        image_train, label_train = next(train_data_gen)
        test_data, teste_valid = next(teste_data_gen)

        train = []
        for img in image_train:
            hist, edg = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
            train.append(hist)

        test = []
        for img in test_data:
            hist, edg = np.histogram(img[:, :, 0], bins=256, range=(0, 256))
            test.append(hist)

        return np.array(train), label_train, np.array(test), teste_valid

    def training_IA(self):
        image_train, label_train, test_data, teste_valid = DeepLearning.get_images(self)

        model = Sequential([
            Dense(512, activation='relu', kernel_regularizer='l2'),
            Dense(256, activation='relu', kernel_regularizer='l2'),
            Dense(128, activation='relu', kernel_regularizer='l2'),
            Dense(64, activation='relu', kernel_regularizer='l2'),
            Dense(32, activation='relu', kernel_regularizer='l2'),
            Dense(18, activation='relu', kernel_regularizer='l2'),
            Dense(9, activation='relu', kernel_regularizer='l2'),
            Dense(5, activation='relu', kernel_regularizer='l2'),
            Dense(len(CLASS_NAMES), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        mdlckpt = tf.keras.callbacks.ModelCheckpoint('model_cpk.h5', monitor='val_loss', verbose=1, mode='min',
                                                     save_best_only=True)

        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)



        history = model.fit(image_train, label_train, epochs=EPOCHS, validation_split=0.4,
                            callbacks=[mdlckpt])

        test_loss, test_acc = model.evaluate(test_data, teste_valid)
        print(test_acc)

        model.summary()
        model.save("model.h5")

        self.matrix_confusion(model, test_data, teste_valid)
        self.plot_result(history)
        pass

    def matrix_confusion(self, model, valid_data_gen, image_valid):
        cm_title = 'Confusion Matrix'
        tick_marks = np.arange(len(CLASS_NAMES))
        Y_pred = model.predict(valid_data_gen)
        y_pred = np.argmax(Y_pred, axis=1)
        rounded_labels = np.argmax(image_valid, axis=1)
        cm = confusion_matrix(rounded_labels, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(cm_title)
        plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
        plt.yticks(tick_marks, CLASS_NAMES)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")

        plt.ylabel('Label Verdadeiro')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        print(cm_title)
        print(cm)
        print('Classification Report')
        print(classification_report(rounded_labels, y_pred, target_names=CLASS_NAMES))
        plt.show()
    pass

    def plot_result(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(history.epoch))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.show()
    pass

    def preditc_IA(self, image_request):
        imgdata = base64.b64decode(image_request) # decodifica a image de base64
        with open("predictimg.png", 'wb') as f: # salva a imagem no diretorio do projeto
            f.write(imgdata)
        # Restaura a imagem utilizando keras, normalizar esses dados
        test_image = tf.keras.preprocessing.image.load_img("predictimg.png", target_size = (IMG_HEIGHT, IMG_WIDTH))
        # test_image = tf.keras.preprocessing.image.img_to_array(test_image) # Converte a imagem para um array
        test_image = test_image - np.mean(test_image, axis=0)  # zero-centering
        test_image = test_image / np.std(test_image, axis=0)  # normalization
        test_image = np.expand_dims(test_image, axis = 0) # expande o array
        model = load_model('model.h5') # restaura o modelo
        predict = model.predict(test_image) # realiza a predição
        print(CLASS_NAMES[np.argmax(predict[0])], predict[0])
        return '{"ferrugem": ' + str(predict[0][0]) + ', "sadia": ' + str(predict[0][1]) + '}'

    def download_tflite(self):
        model = tf.keras.models.load_model('model.h5')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)
        pass

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(np.array(img,np.int32))

        predicted_label = np.argmax(predictions_array)
        if predicted_label == np.argmax(true_label):
            color = 'green'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                             100 * np.max(predictions_array),
                                             CLASS_NAMES[np.argmax(true_label)]),
                   color=color)

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        tick_marks = np.arange(len(CLASS_NAMES))
        plt.xticks(tick_marks, CLASS_NAMES)
        plt.yticks([])
        thisplot = plt.bar(range(2), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[np.argmax(true_label)].set_color('green')

    def gerar_dados(self):
        model = tf.keras.models.load_model('model.h5')

        teste_data_gen = ImageDataGenerator().flow_from_directory(directory=str(test_dir2), batch_size=16,
                                                                  target_size=(
                                                                      IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode="categorical",
                                                                  shuffle=False
                                                                  )
        test_data, test_labels = next(teste_data_gen)

        test = []
        for img in test_data:
            hist, edg = np.histogram(img[:, :, 2], bins=256, range=(0, 256))
            test.append(hist)

        predictions = model.predict(np.array(test))
        num_rows = 4
        num_cols = 4
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions, test_labels, test_data)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, predictions, test_labels)
        plt.show()


ia = DeepLearning()
# ia.training_IA()
ia.gerar_dados()
# train_data_gen, test_data = ia.get_images()
# image_train, label_train = next(train_data_gen)
# test_data, teste_valid = next(test_data)
# ia.training_IA()
# ia.download_tflite()
# test_data = test_data - np.mean(test_data, axis=0)  # zero-centering
# test_data = test_data / np.std(test_data, axis=0)  # normalization
# img = cv2.imread('./train/ferrugem/ferrugem_tan_049.png')
#
# img_to_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
# hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
# cv2.imwrite('result.jpg', hist_equalization_result)
#
# color = ('b','g','r')
# for channel,col in enumerate(color):
#     histr = cv2.calcHist([img],[channel],None,[256],[0,256])
#     plt.subplot(1, 4, 1)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title('ferrugem normal')
#
# color = ('b','g','r')
# for channel,col in enumerate(color):
#     histr = cv2.calcHist([hist_equalization_result],[channel],None,[256],[0,256])
#     plt.subplot(1, 4, 2)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title('ferrugme equal')
#
# img = cv2.imread('./train/sadia/sadia_091.png')
# color = ('b','g','r')
# for channel,col in enumerate(color):
#     histr = cv2.calcHist([img],[channel],None,[256],[0,256])
#     plt.subplot(1, 4, 3)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title('sadia normal')
#
# img_to_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
# hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
#
# color = ('b','g','r')
# for channel,col in enumerate(color):
#     histr = cv2.calcHist([hist_equalization_result],[channel],None,[256],[0,256])
#     plt.subplot(1, 4, 4)
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title('sadia equal')
#
# plt.show()

# Faz a varredura do diretório imagens buscando arquivos JPG, JPEG e PNG.
# diretorio = 'train/ferrugem'
# arquivos = os.listdir(diretorio)
# train_data = []
# for a in arquivos:
#     if a.lower().endswith('.jpg') or a.lower().endswith('.png') or a.lower().endswith('.jpeg'):
#         imgC = cv2.imread(diretorio + '/' + a)
#         x = img_to_array(imgC)
#         x = x.reshape((1,) + x.shape)
#         train_data.append(imgC)
#     cv2.waitKey(0)