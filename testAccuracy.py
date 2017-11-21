import random
import asyncio
import numpy as np
import pandas as pd
import sys
import personalities

# Use actual location of the training data CSV file
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

userid_to_emotion_dictionary = personalities.personalities.run(train_profile_df,df_likes,test_profile_df,df_likes_test)

RMSE = [0,0,0,0,0]

for index,row in test_profile_df.iterrows():
    userid = row['userid']
    emotion_array = userid_to_emotion_dictionary[userid]
    err_ope = row['ope'] - emotion_array[1]
    err_con = row['con'] - emotion_array[2]
    err_ext = row['ext'] - emotion_array[3]
    err_agr = row['agr'] - emotion_array[4]
    err_neu = row['neu'] - emotion_array[5]
    RMSE[0] += err_ope * err_ope
    RMSE[1] += err_con * err_con
    RMSE[2] += err_ext * err_ext
    RMSE[3] += err_agr * err_agr
    RMSE[4] += err_neu * err_neu

RMSE[0]/=n
RMSE[1]/=n
RMSE[2]/=n
RMSE[3]/=n
RMSE[4]/=n

print(np.sqrt(RMSE))