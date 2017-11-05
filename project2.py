import os
import sys
import pandas as pd
# Take these values from command line
from Gender_By_Likes import Gender_By_Likes
from Gender_By_Status import Gender_By_Status
from personalities import personalities

if sys.argv.__len__() != 3:
    print("ERROR: please specify test data directory and output directory in command line arguments")
    exit(-1)

test_data_directory = sys.argv[1]
output_directory = sys.argv[2]

# creating a directory named output_directory
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# Use actual location of the training data CSV file
training_data = "/data/training/profile/profile.csv"
df = pd.read_csv(training_data)

test_data = test_data_directory + '/profile/profile.csv'
testdf = pd.read_csv(test_data)

data_gender = df.loc[:, ['gender']]
data_Age = df.loc[:, ['age']]
data_Personality = df.loc[:, ['ope', 'con', 'ext', 'agr', 'neu']]

# Counting age ranges
age_grp_xx_24 = 0
age_grp_25_34 = 0
age_grp_35_49 = 0
age_grp_50_xx = 0

for age in data_Age['age']:
    if (age >= 00 and age < 25):
        age_grp_xx_24 = age_grp_xx_24 + 1
    if (age >= 25 and age < 35):
        age_grp_25_34 = age_grp_25_34 + 1
    if (age >= 35 and age < 50):
        age_grp_35_49 = age_grp_35_49 + 1
    if (age >= 50):
        age_grp_50_xx = age_grp_50_xx + 1

print("Age group age_grp_xx_24: %d" % age_grp_xx_24)
print("Age group age_grp_25_34: %d" % age_grp_25_34)
print("Age group age_grp_35_49: %d" % age_grp_35_49)
print("Age group age_grp_50_xx: %d" % age_grp_50_xx)

max_age = max(age_grp_xx_24, age_grp_25_34, age_grp_35_49, age_grp_50_xx)
if (max_age == age_grp_xx_24):
    age_grp = "xx_24"
elif (max_age == age_grp_25_34):
    age_grp = "25_34"
elif (max_age == age_grp_35_49):
    age_grp = "35_49"
elif (max_age == age_grp_50_xx):
    age_grp = "50_xx"

df_likes = pd.read_csv(r'/data/training/relation/relation.csv')
df_likes_test = pd.read_csv(test_data_directory + '/relation/relation.csv')

#Getting userid to gender dictionary predicted from user status
userId_to_GenderByStatus = Gender_By_Status.run(df, testdf, test_data_directory)

#Getting userid to gender dictionary predicted from user likes
# userId_to_GenderByLikes = Gender_By_Likes.run(df,df_likes,testdf,df_likes_test)

#Getting userid to array of emotions dictionary predicted from user likes
userid_to_emotions_dictionary = personalities.run(df, df_likes, testdf, df_likes_test)

print("Saving output")

count_gender_match = 0
count_gender_mismatch = 0

# Generating XML files

correct_count = 0
wrong_count = 0

for row in testdf.loc[:, ['userid']].iterrows():
    userid = getattr(row[1], 'userid')

    gender_status = userId_to_GenderByStatus[userid]

    # gender_likes = userId_to_GenderByLikes[userid]

    final_gender = gender_status

    emotions_array = userid_to_emotions_dictionary[userid]
    ope = emotions_array[1]
    con = emotions_array[2]
    ext = emotions_array[3]
    agr = emotions_array[4]
    neu = emotions_array[5]

    # Use this when test data doesn't contain labels
    xml = '<user id="{0}"\nage_group="{1}"\ngender="{2}"\nextrovert="{5}"\nneurotic="{7}"\nagreeable="{6}"\nconscientious="{4}"\nopen="{3}"\n/>'.format(
        userid, age_grp, final_gender, ope, con, ext, agr, neu)
    text_file = open('{0}/{1}.xml'.format(output_directory, userid), "w")

    text_file.write(xml)
    text_file.close()

print("Gender match: %d" % count_gender_match)
print("Gender mismatch: %d" % count_gender_mismatch)

# print("Accuracy after combination: %f" % ((correct_count)/(correct_count+wrong_count)))
