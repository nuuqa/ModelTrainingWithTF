# Tämän koodin luomisessa on käytetty apuna TensorFlow:n dokumentaatiota,
# https://www.kaggle.com/code/maosama/dog-cat-90-accuracy-no-transfer-learning osoitteesta löytyvää tietoa/koodia,
# sekä ChatGPT.

#Import
import tensorflow as tf
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Määritä kansioiden sijainnit
# Koulutus
dataDir = Path("C:/Temp/Data/PetImages")
image_count = len(list(dataDir.glob('*/*.jpg')))
print("Koulutuskuvien määrä: ", image_count)

# Testaus
testDataDir = Path("C:/Temp/Data/TestData")
image_count = len(list(testDataDir.glob('*/*.jpg')))
print("Testikuvien määrä: ", image_count)

# Kuvan koot ja erän koko.
batch_size = 32  # => 32 kuvaa per batch
img_height = 128
img_width = 128

# Koulutus/Validointi ImageDataGenerator
dataAug = ImageDataGenerator(
    rescale=1./255,  # normalisoidaan arvot välille 0 - 1
    validation_split=0.1,  # 10% datasta käytetään validointiin
    zoom_range=0.2,  # Zoomia
    rotation_range=40, # Pyöritellään
    horizontal_flip=True
)

# Testauksen ImageDataGenerator
testDataAug = ImageDataGenerator(
    rescale=1./255  # Sama kuin koulutuksessa, 0 - 1
)

# Koulutus data
trainDataSet = dataAug.flow_from_directory(
    dataDir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validointi data
valDataSet = dataAug.flow_from_directory(
    dataDir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Testaus data
testDataSet = testDataAug.flow_from_directory(
    testDataDir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Koulutus
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding="SAME", activation='relu', input_shape=(img_height, img_width, 3)),  # 3 => R, G, B
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="SAME", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding="SAME", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), padding="SAME", activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 0 tai 1, kissa tai koira.
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

stopEarly = EarlyStopping(monitor="val_loss", patience=2)  # Jos ei parane kahden Epochin jälkeen => katkaistaan koulutus.

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(trainDataSet,
          epochs=50,
          validation_data=valDataSet,
          callbacks=[stopEarly, tensorboard_callback])

test_loss, test_accuracy = model.evaluate(testDataSet)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
model.save("CatAndDogs.h5")
