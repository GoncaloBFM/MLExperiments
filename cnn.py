import numpy
from keras import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Dense
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Convolution2D(32, 3, 3, border_mode="same", activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation="relu"))
classifier.add(Dense(output_dim=1, activation="sigmoid"))
classifier.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    './data/animals/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    './data/animals/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    train_generator,
    steps_per_epoch=8000,
    validation_steps=2000,
    epochs=25,
    validation_data=test_generator,
    use_multiprocessing=True,
)

# MAKE SINGLE PREDICTION
to_predict = image.array_to_img(image.load_img("image path", target_size=(64, 64)))
print(classifier.predict(numpy.expand_dims(to_predict, axis=0)))
print(train_generator.class_indices)
