from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from cnn_image_data_processor import group_images_by_category, split_dataset


# dimensions of our images.
img_width, img_height = 150, 150

# Splitting data into training & validation folders
train_dir, validation_dir = split_dataset('training_data/image', 'organized_images', 0.8)

# Grouping training data by category
group_images_by_category(train_dir, 'final_organized_images/training_split/by_gender', 'gender', 'training_data/profile/profile.csv')

# Grouping validation data by category
group_images_by_category(validation_dir, 'final_organized_images/validation_split/by_gender', 'gender', 'training_data/profile/profile.csv')

train_data_dir = 'final_organized_images/training_split/by_gender'
validation_data_dir = 'final_organized_images/validation_split/by_gender'


nb_train_samples = 7600 # 9500 * 0.8
nb_validation_samples = 1900 # 9500 * 0.2

epochs = 50
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# steps_per_epoch should be (number of training images total / batch_size) => 7600/100 = 76
# validation_steps should be (number of validation images total / batch_size) => 1900/100 = 19

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model.save_weights('trained_cnn_image_classifier.h5')
