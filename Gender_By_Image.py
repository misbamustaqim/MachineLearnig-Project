from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image
import numpy as np
from skimage import color, exposure, transform
from skimage import io
import glob

(img_width, img_height) = (150, 150)


def load_cnn_model():
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

	# Load the model/weights
	model.load_weights('/home/itadmin/MachineLearnig-Project/trained_cnn_image_classifier.h5')
	model.compile(loss='binary_crossentropy', optimizer='rmsprop',
		      metrics=['accuracy'])
	return model
    
class Gender_By_Image:

    
    def run(test_profile_df, test_data_directory):
        print ('Predicting gender from image..')
        userId_to_GenderByImage = dict()

        print ('Loading trained model..')
        model = load_cnn_model()

        for (index, row) in test_profile_df.iterrows():
            data_userid = row['userid']
            image_path = test_data_directory + '/image/' + data_userid + '.jpg'
            #print (image_path)
            img = load_img(image_path, False, target_size=(img_width, img_height))
            x = img_to_array(img)
            x = x / 255
            x = np.expand_dims(x, axis=0)
            # predicting gender
            predicted_gender = model.predict_classes(x)
            # prob = model.predict_proba(x)
            if (predicted_gender[0] == 1.0):
                userId_to_GenderByImage[data_userid] = '1.0'
            else:
                userId_to_GenderByImage[data_userid] = '0.0'
            #userId_to_GenderByImage[data_userid] = predicted_gender[0]

        print ('Completed gender prediction from images')

        return userId_to_GenderByImage
			

    
