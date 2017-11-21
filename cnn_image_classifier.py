from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from skimage import color, exposure, transform
from skimage import io
import glob



IMG_SIZE = 150


def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img


# dimensions of our images.
img_width, img_height = 150, 150

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

# Starts here
model.load_weights('trained_cnn_image_classifier.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Predict category for the test data
file_list = glob.glob('/home/itadmin/tcss555 (2)/public-test-data/image/*.jpg')


for index, elem in enumerate(file_list):
    img = load_img(elem,False,target_size=(img_width,img_height))
    x = img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    prob = model.predict_proba(x)

    # if (preds == 1.0):
    #     gender = 'Female'
    # elif (preds == 0.0):
    #     gender = 'Male'
    # else:
    #     gender = 'Unknown'

    print ('image: {0}, predicted gender: {1}, probability: {2}'.format(elem, preds, prob))
