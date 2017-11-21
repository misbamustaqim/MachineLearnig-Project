import os
import glob
import shutil
import pandas as pd
import numpy as np
import sys

def create_directory(dir_name):
	os.makedirs(dir_name, exist_ok=True)

def group_images_by_category(source_folder, destination_folder, category, mapping_file_location):

	if not source_folder:
		print ('Image source_folder is required to continue')
		return
	if not destination_folder:
		print ('Image destination_folder is required to continue')
		return
	if not mapping_file_location:
		print ('Image to category mapping file is required to continue')
		return
	if not category:
		print ('group by category is required to continue')
		return

	df = pd.read_csv(mapping_file_location, header=0)

	for index, row in df.iterrows():
		# create directory for this class of images
		create_directory(destination_folder + "/"+ str(row[category]))

		# Assuming that all the images have .jpg extension
		src_img_location = source_folder + "/" + row['userid'] + ".jpg"
		dest_img_location = destination_folder + "/" + str(row[category])

		#print ('Source image location: {0}, Copying this file to {1}'.format(src_img_location, dest_img_location))
		try:
			shutil.copy2(src_img_location, dest_img_location)
		except Exception as e:
			print ('Error copying %s to %s: %s' % (src_img_location, dest_img_location, e))
			pass

	# done
	print ('Images are organized into folders per each category: ' + category)

def split_dataset(source_directory, target_directory, split_ratio, file_extension='*.jpg'):

	# Directory should be the dataset directory.
	if not os.path.exists(source_directory):
		return 0

	file_list = glob.glob(source_directory + '/*.jpg')

	train_dir = os.path.abspath(target_directory + '/training_split')
	validation_dir = os.path.abspath(target_directory + '/validation_split')

	create_directory(train_dir)
	create_directory(validation_dir)

	train_images = []
	validation_images = []
	random_set = np.random.permutation(len(file_list))
	train_list = random_set[:round(len(random_set)*split_ratio)]
	test_list = random_set[-(len(file_list) - len(train_list))::]

	for index in train_list:
		train_images.append(file_list[index])
	for index in test_list:
		validation_images.append(file_list[index])

	print ('Total images..  ' + str(len(file_list)))
	print ('Training images..  ' + str(len(train_images)))
	print ('Validation images.. ' + str(len(validation_images)))

	print ('Copying training data into folder: ' + train_dir)
	for file in train_images:
		try:
			shutil.copy2(file, train_dir)
		except Exception as e:
			print ('Error copying %s to %s: %s' % (src_img_location, dest_img_location, e))
			pass

	print ('Copying validation data into folder: ' + validation_dir)
	for file in validation_images:
		try:
			shutil.copy2(file, validation_dir)
		except Exception as e:
			print ('Error copying %s to %s: %s' % (src_img_location, dest_img_location, e))
			pass
	return train_dir, validation_dir

# def main():
#
# 	# Splitting data into training & validation folders
# 	train_dir, validation_dir = split_dataset('training_data/image', 'organized_images', 0.8)
#
# 	# Grouping training data by category
# 	group_images_by_category(train_dir, 'final_organized_images/training_split/by_gender', 'gender', 'training_data/profile/profile.csv')
#
# 	# Grouping validation data by category
# 	group_images_by_category(validation_dir, 'final_organized_images/validation_split/by_gender', 'gender', 'training_data/profile/profile.csv')
#
# if __name__ == "__main__":
#     main()
