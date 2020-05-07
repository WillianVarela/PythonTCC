import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import IPython.display as display
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


AUTOTUNE = tf.data.experimental.AUTOTUNE

# pega diretorio das imagens
train_dir = pathlib.Path('train/')
valid_dir = pathlib.Path('validation/')
# Define as classes
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])
print(CLASS_NAMES)

# Pega lista de imagens
train_count = len(list(train_dir.glob('*/*.jpg')))
list_train_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
valid_count = len(list(valid_dir.glob('*/*.jpg')))
list_valid_ds = tf.data.Dataset.list_files(str(valid_dir/'*/*'))
print('total train imagens', train_count)
print('total valid imagens', train_count)

# Defina Variaveis das iamgens
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
STEPS_PER_EPOCH = np.ceil(train_count/BATCH_SIZE)


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Cria um conjunto de dados pares
labeled_train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_valid_ds = list_valid_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in labeled_ds.take(1):
#    print("Image shape: ", image.numpy().shape) # Image shape:  (224, 224, 3)
#    print("Label: ", label.numpy()) # Label:  [ True False]

# Inspeciona Lote dde imagens


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')
    plt.show()


# Preparando para treinar
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(labeled_train_ds)
image_train, label_train = next(iter(train_ds))

valid_ds = prepare_for_training(labeled_valid_ds)
image_valid, label_valid = next(iter(train_ds))

# mostra imagens da base
#show_batch(image_batch.numpy(), label_batch.numpy())

# Cria uma rede convolucional
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# ------------------------------------------------------------------------------------------

# Exibe arquitetura do modelo formado
model.summary()
# ------------------------------------------------------------------------------------------

# Adiciona a ultima camada de densidade
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))
# ------------------------------------------------------------------------------------------

# Exibe arquitetura final do modelo formado
model.summary()
# ------------------------------------------------------------------------------------------

# Compilando e treinando o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(image_train, label_train, epochs=10,
                    validation_data=(image_valid, label_valid))
# ------------------------------------------------------------------------------------------

# Avaliacao do modelo
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(image_valid,  label_valid, verbose=2)
# ------------------------------------------------------------------------------------------

print(test_acc)