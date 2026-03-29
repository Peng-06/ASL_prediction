import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = os.path.join('.','asl_alphabet_train','asl')
IMG_SIZE = 200
BATCH_SIZE = 32
EPOCHS = 20

data_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
train_gen = data_gen.flow_from_directory(
    TRAIN_DIR, target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE, class_mode = 'sparse', subset = 'training'
)

val_gen = data_gen.flow_from_directory(
    TRAIN_DIR, target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH_SIZE, class_mode = 'sparse', subset = 'validation'
)

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu', input_shape = (IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2))

    layers.Flatten(),
    layers.Dense(512,activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense (29,activation = 'relu')
]
)

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

model.fit(train_gen, validation_data = val_gen, epochs = EPOCHS)
model.save(os.path.join('..','ASL.h5'))

print("Saved ASL.h5 in ASL_FINAL")