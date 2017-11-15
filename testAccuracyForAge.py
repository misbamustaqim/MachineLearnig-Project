import random
import numpy as np
import pandas as pd

# Use actual location of the training data CSV file
from Age_By_Likes import Age_By_Likes

training_data = "/data/training/profile/profile.csv"
data_FBUsers_train = pd.read_csv(training_data)
train_profile_df = data_FBUsers_train

df_likes = pd.read_csv(r'/data/training/relation/relation.csv')

# Splitting the data into training instances and test instances
n = 900
all_Ids = np.arange(len(train_profile_df))
random.shuffle(all_Ids)
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
test_profile_df = data_FBUsers_train.loc[test_Ids, :]
train_profile_df = data_FBUsers_train.loc[train_Ids, :]
df_likes_test = df_likes

userid_to_age = Age_By_Likes.run(train_profile_df,df_likes,test_profile_df,df_likes_test)

RMSE = [0,0,0,0,0]

countMatch = 0.0
countMismatch = 0.0

for index,row in test_profile_df.iterrows():
    userid = row['userid']
    predictedAge = userid_to_age[userid]
    age = row['age']
    actualAge = ""

    if (age >= 00 and age < 25):
        actualAge = "xx_24"
    if (age >= 25 and age < 35):
        actualAge = "25_34"
    if (age >= 35 and age < 50):
        actualAge = "35_49"
    if (age >= 50):
        actualAge = "50_xx"

    if(actualAge == predictedAge):
        countMatch+=1.0
    else:
        countMismatch+=1.0


print(countMatch)
print(countMismatch)

print("Accuracy:")
print(countMatch/ (countMatch + countMismatch))