import random
import numpy as np
import pandas as pd
import sys
import csv
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Take these values from command line
test_data_directory=sys.argv[1]
output_directory=sys.argv[2]

# Use actual location of the training data CSV file
training_data= "/data/training/profile/profile.csv"
df = pd.read_csv(training_data)

test_data = test_data_directory + '/profile/profile.csv'
testdf = pd.read_csv(test_data)

data_gender = df.loc[:,['gender']]
data_Age = df.loc[:,['age']]
data_Personality = df.loc[:,['ope', 'con','ext','agr','neu']]

# Counting age ranges
age_grp_xx_24 = 0
age_grp_25_34 = 0
age_grp_35_49 = 0
age_grp_50_xx = 0

for age in data_Age['age']:
	if(age>=00 and age <25):
		age_grp_xx_24 = age_grp_xx_24 + 1
	if(age>=25 and age <35):
		age_grp_25_34 = age_grp_25_34 + 1
	if(age>=35 and age < 50):
		age_grp_35_49 = age_grp_35_49 + 1
	if(age>=50):
		age_grp_50_xx = age_grp_50_xx + 1

print("Age group age_grp_xx_24: %d" % age_grp_xx_24)
print("Age group age_grp_25_34: %d" % age_grp_25_34)
print("Age group age_grp_35_49: %d" % age_grp_35_49)
print("Age group age_grp_50_xx: %d" % age_grp_50_xx)

max_age = max(age_grp_xx_24, age_grp_25_34,age_grp_35_49,age_grp_50_xx)
if (max_age == age_grp_xx_24):
    age_grp = "xx_24"
elif (max_age == age_grp_25_34):
    age_grp = "25_34"
elif (max_age == age_grp_35_49):
    age_grp = "35_49"
elif (max_age == age_grp_50_xx):
    age_grp = "50_xx"

# Getting averages for all the emotions
ave_ope = np.mean(data_Personality['ope'])
ave_con = np.mean(data_Personality['con'])
ave_ext = np.mean(data_Personality['ext'])
ave_agr = np.mean(data_Personality['agr'])
ave_neu = np.mean(data_Personality['neu'])

print("Average of ope: %f" % ave_ope)
print("Average of con: %f" % ave_con)
print("Average of ext: %f" % ave_ext)
print("Average of agr: %f" % ave_agr)
print("Average of neu: %f" % ave_neu)

print("Predicting Gender from status...")

for index, row in df.iterrows():
    data_userid = row['userid']
    userid_to_txt = open('/data/training/text/' +data_userid+'.txt','r',errors='ignore')
    data_status = userid_to_txt.read()
    df.set_value(index,'Status',data_status)
    userid_to_txt.close()
df.to_csv('New_profile.csv',sep=',')

for index, row in testdf.iterrows():
    data_userid = row['userid']
    userid_to_txt = open(test_data_directory + '/text/' +data_userid+'.txt','r',errors='ignore')
    data_status = userid_to_txt.read()
    testdf.set_value(index,'Status',data_status)
    userid_to_txt.close()
testdf.to_csv('New_profile_test.csv',sep=',')

# Reading the data into a dataframe and selecting the columns we need

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
 csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
 df = pd.read_csv(utf_8_encoder('New_profile.csv'))
#print(df)
#,header=None,usecols=['gender'])
data_FBUsers = df.loc[:,['userid','gender','age','Status']]
#print(data_FBUsers)

#reading training data into a dataframe
def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
 csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
 testdf = pd.read_csv(utf_8_encoder('New_profile_test.csv'))
data_FBUsers_test = testdf.loc[:,['userid','gender','age','Status']]

#Getting training and test data
data_test = data_FBUsers.loc[np.arange(len(data_FBUsers_test)), :]
data_train = data_FBUsers.loc[np.arange(len(data_FBUsers)), :]

# Splitting the data into training instances and test instances
#n = 900
#all_Ids = np.arange(len(data_FBUsers))
#random.shuffle(all_Ids)
#test_Ids = all_Ids[0:n]
#train_Ids = all_Ids[n:]
#data_test = data_FBUsers.loc[test_Ids, :]
#data_train = data_FBUsers.loc[train_Ids, :]


actual_gender = dict(zip(data_test['userid'], data_test['gender']))
# Training a Naive Bayes model for Gender prediction
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['Status'])
y_train = data_train['gender']
clf = MultinomialNB()
clf.fit(X_train, data_train['gender'])

# Testing the Naive Bayes model for Gender prediction
X_test = count_vect.transform(data_test['Status'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

#===============================================================MISBA============================================================================================
print("Predicted gender using status")

print("Predicting gender using likes")

# Reporting on classification performance of Gender prediction
#print("Accuracy of Gender prediction: %.2f" % accuracy_score(y_test,y_predicted))

profile_dict = dict(zip(data_train['userid'], data_train['gender']))
LikeIdGender = dict()
LikeID_to_Weight = dict()
userID_to_totalWeight = dict()

male_likes = 0
female_likes = 0

df_likes = pd.read_csv(r'/data/training/relation/relation.csv')
df_likes_test = pd.read_csv(test_data_directory + '/relation/relation.csv')

#train data
for row in df_likes.iterrows():

	userid = row[1]['userid']
	likeId = row[1]['like_id']

	#if "message" in dictionary:
    	#	data = dictionary["message"]

	gender = profile_dict.get(userid)
	if gender==0:
		if likeId not in LikeIdGender:
			LikeIdGender[likeId] = [0,0]
		LikeIdGender[likeId][0] = LikeIdGender[likeId][0] + 1
		male_likes = male_likes + 1

	else:
		if likeId not in LikeIdGender:
			LikeIdGender[likeId] = [0,0]
		LikeIdGender[likeId][1] = LikeIdGender[likeId][1] + 1
		female_likes = female_likes + 1


for key, value in LikeIdGender.items():
	Difference = LikeIdGender[key][0] - LikeIdGender[key][1]
	Sum = LikeIdGender[key][0] + LikeIdGender[key][1]
	square=Difference**(2)
	weightOfPage = square / Sum
	if(Difference > 0):
		LikeID_to_Weight[key] = (1/male_likes)
	else:
		if(weightOfPage <= 2.0):
			continue
		LikeID_to_Weight[key] = - (1/female_likes)


#prediction:

page_not_liked = 0

for row in df_likes_test.iterrows():
	userid = row[1]['userid']
	likeId = row[1]['like_id']

	if userid not in userID_to_totalWeight:
		userID_to_totalWeight[userid] = 0

	if(likeId not in LikeID_to_Weight):
		page_not_liked = page_not_liked +1
		#weight = (1.0/male_likes)
		continue
	else:
		weight = LikeID_to_Weight[likeId]

	userID_to_totalWeight[userid] = userID_to_totalWeight[userid] + weight

#print(LikeID_to_Weight)

correct_count = 0
wrong_count = 0
zero_weight = 0
predicted_male = 0
predocted_female = 0
right_predicted_male = 0
right_predicted_female = 0

useridToPredictedGender = dict()

for key, value in userID_to_totalWeight.items():
	userid = key

	if(value > 0):
		predicted_gender = 0.0
		predicted_male = predicted_male + 1
	else:
		predicted_gender = 1.0
		predocted_female = predocted_female + 1
	useridToPredictedGender[userid] = predicted_gender

	if(value == 0):
		zero_weight = zero_weight + 1
	
	if(userid in actual_gender):
		if(actual_gender[userid] == predicted_gender):
			correct_count+=1
		else:
			wrong_count+=1
		
print("ZeroWeight: %d" % zero_weight)
#print("Correct: %d" % correct_count)
#print("Wrong: %d" % wrong_count)

print("Predicted Male: %d" % predicted_male)
print("Predicted Female: %d" % predocted_female)

print("Total Male likes: %d" % male_likes)
print("Total Female likes: %d" % female_likes)
print("Pages not liked: %d" % page_not_liked)

#print("Accuracy: %f" % ((correct_count)/(correct_count+wrong_count)))


print("Saving output")

count_gender_match = 0
count_gender_mismatch = 0

# Generating XML files
#creating a directory named output_directory
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

correct_count = 0
wrong_count = 0
i=0

for index, row in data_FBUsers_test.iterrows():
    # Use this when test data contain labels
    #xml = '<user id="{0}" age_group="{1}" gender="{2}" openness="{3}" conscientious="{4}" extrovert="{5}" agreeable="{6}" neurotic="{7}"/>'.format(row['userid'],row['age'],gender,row['ope'],row['con'],row['ext'],row['agr'],row['neu'] )
    #text_file = open(output_directory+"/"+row['userid']+".xml", "w")

	userid = row[0]
	gender_status = y_predicted[i]
	i+=1

	weight = userID_to_totalWeight[userid]
	if(weight > 0):
		gender_likes = 0.0
	else:
		gender_likes = 1.0

	if(weight == 0):
		gender_likes = gender_status

	if(gender_status == gender_likes):
		final_gender = gender_status
		count_gender_match = count_gender_match +1
	else:
		final_gender = gender_likes
		count_gender_mismatch = count_gender_mismatch + 1

	#if(actual_gender[userid] == final_gender):
	#	correct_count+=1
	#else:
	#	wrong_count+=1

    # Use this when test data doesn't contain labels
	xml = '<user id="{0}"\nage_group="{1}"\ngender="{2}"\nextrovert="{5}"\nneurotic="{7}"\nagreeable="{6}"\nconscientious="{4}"\nopen="{3}"\n/>'.format(row[0],age_grp,final_gender,ave_ope,ave_con,ave_ext,ave_agr,ave_neu )
	#stext_file = open('{0}/{1}.xml'.format(output_directory, row[0]), "w")
	#xml = '<user id="{0}" age_group="{1}" gender="{2}" openness="{3}" conscientious="{4}" extrovert="{5}" agreeable="{6}" neurotic="{7}"/>'.format(row[0],age_grp,gender,ave_ope,ave_con,ave_ext,ave_agr,ave_neu )
	text_file = open('{0}/{1}.xml'.format(output_directory, row[0]), "w")
	#print (output_directory)
	#print (row[1])
	text_file.write(xml)
	text_file.close()

print("Gender match: %d" % count_gender_match) 
print("Gender mismatch: %d" % count_gender_mismatch) 

#print("Accuracy after combination: %f" % ((correct_count)/(correct_count+wrong_count)))







