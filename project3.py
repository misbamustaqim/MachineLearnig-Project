import os
import sys
import pandas as pd
import numpy as np
import nltk
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#nltk.download()

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


#########################################Predicting_Emotions_from_LIWC_Features###########################
print("Predicting emotions from Status...")

#training_profile_data_emotions = "/data/training/profile/profile.csv"
#df_profile_emotions = pd.read_csv(training_profile_data_emotions)
#test_data_emotions = test_data_directory + '/profile/profile.csv'
#testdf_profile_emotions = pd.read_csv(test_data_emotions)

df_LIWCFeature_training = pd.read_csv(r'/data/training/LIWC/LIWC.csv')
df_LIWCFeature_test=pd.read_csv(test_data_directory+'/LIWC/LIWC.csv')

df_LIWCFeature_training.columns = [x.lower() for x in df_LIWCFeature_training.columns]
df_LIWCFeature_merge_profile_training = pd.merge(df_LIWCFeature_training,df,on='userid')
df_LIWCFeature_merge_profile_training.drop(['Unnamed: 0','age','gender'],axis=1, inplace=True)

df_LIWCFeature_test.columns = [x.lower() for x in df_LIWCFeature_test.columns]
df_LIWCFeature_merge_profile_test = pd.merge(df_LIWCFeature_test,testdf,on='userid')
df_LIWCFeature_merge_profile_test.drop(['Unnamed: 0','age','gender'],axis=1, inplace=True)

# Preparing the train and test data
big5 = ['ope','ext','con','agr','neu']
LIWC_features = [x for x in df_LIWCFeature_merge_profile_training.columns.tolist()[:] if not x in big5]
LIWC_features.remove('userid')
X_train_emotions = df_LIWCFeature_merge_profile_training[LIWC_features]
y_train_ope = df_LIWCFeature_merge_profile_training.ope #selecting ope as the target
y_train_con = df_LIWCFeature_merge_profile_training.con #selecting con as the target
y_train_ext = df_LIWCFeature_merge_profile_training.ext #selecting extrovert as the target
y_train_agr = df_LIWCFeature_merge_profile_training.agr #selecting agr as the target
y_train_neu = df_LIWCFeature_merge_profile_training.neu #selecting neurotic as the target

LIWC_features = [x for x in df_LIWCFeature_merge_profile_test.columns.tolist()[:] if not x in big5]
LIWC_features.remove('userid')
X_test_emotions = df_LIWCFeature_merge_profile_test[LIWC_features]

# Training and evaluating a linear regression model
linreg_ope = LinearRegression()
linreg_con = LinearRegression()
linreg_ext = LinearRegression()
linreg_agr = LinearRegression()
linreg_neu = LinearRegression()

linreg_ope.fit(X_train_emotions,y_train_ope)
linreg_con.fit(X_train_emotions,y_train_con)
linreg_ext.fit(X_train_emotions,y_train_ext)
linreg_agr.fit(X_train_emotions,y_train_agr)
linreg_neu.fit(X_train_emotions,y_train_neu)
# Evaluating the model
y_predict_emotions_ope = linreg_ope.predict(X_test_emotions)
y_predict_emotions_con = linreg_con.predict(X_test_emotions)
y_predict_emotions_ext = linreg_ext.predict(X_test_emotions)
y_predict_emotions_agr = linreg_agr.predict(X_test_emotions)
y_predict_emotions_neu = linreg_neu.predict(X_test_emotions)

print("Average of Openness Emotions: ",np.mean(y_predict_emotions_ope))
print("Average of Conscientiousness Emotions: ",np.mean(y_predict_emotions_con))
print("Average of Extroversion Emotions: ",np.mean(y_predict_emotions_ext))
print("Average of Agreeableness Emotions: ",np.mean(y_predict_emotions_agr))
print("Average of Emotional Stability: ",np.mean(y_predict_emotions_neu))

userid_to_emotion_ope_dictionary = dict()
userid_to_emotion_con_dictionary = dict()
userid_to_emotion_ext_dictionary = dict()
userid_to_emotion_agr_dictionary = dict()
userid_to_emotion_neu_dictionary = dict()

for index, row in testdf.iterrows():
    userid = getattr(row , 'userid')
    userid_to_emotion_ope_dictionary[userid]  = y_predict_emotions_ope[index]
    userid_to_emotion_con_dictionary[userid]  = y_predict_emotions_con[index]
    userid_to_emotion_ext_dictionary[userid]  = y_predict_emotions_ext[index]
    userid_to_emotion_agr_dictionary[userid]  = y_predict_emotions_agr[index]
    userid_to_emotion_neu_dictionary[userid]  = y_predict_emotions_neu[index]


##########################################Predicting_Gender_from_Status#############################

for index, row in df.iterrows():
    data_userid = row['userid']
    userid_to_txt = open('/data/training/text/' + data_userid + '.txt', 'r', errors='ignore')
    #userid_to_txt = open('/data/training/text/' + data_userid + '.txt', 'r', errors='ignore')
    data_status = userid_to_txt.read()
    df.set_value(index, 'Status', data_status)
    userid_to_txt.close()

for (index, row) in testdf.iterrows():
    data_userid = row['userid']
    userid_to_txt = open(test_data_directory + '/text/' + data_userid + '.txt', 'r', errors='ignore')
    data_status = userid_to_txt.read()
    testdf.set_value(index, 'Status', data_status)
    userid_to_txt.close()

df.loc[df['age'] < 25,'age_grp'] = "XX_24"
df.loc[(df['age'] >= 25) & (df['age'] < 35),'age_grp'] = "25_34"
df.loc[(df['age'] >=35) & (df['age'] < 50), 'age_grp'] = "35_49"
df.loc[df['age'] >= 50, 'age_grp'] = "50_XX"

data_FBUsers_train = df.loc[:, ['userid', 'gender', 'Status', 'age_grp']]
data_FBUsers_test = testdf.loc[:, ['userid', 'gender', 'Status', 'age_grp']]

##################Splitting the data into 8000 training and 1500 test instances##################

#n=1500
#all_Ids = np.arange(len(data_FBUsers))
#random.shuffle(all_Ids)
#test_Ids = all_Ids[0:n]
#train_Ids = all_Ids[n:]
#data_test = data_FBUsers.loc[test_Ids,:]
#data_train = data_FBUsers.loc[train_Ids,:]

################################# Getting training and test data###################################

test_data = data_FBUsers_test.loc[np.arange(len(data_FBUsers_test)), :]
train_data = data_FBUsers_train.loc[np.arange(len(data_FBUsers_train)), :]

#################################Predicting Gender from Status######################################

print("Predicting Gender by Status....")
text_feature = ['Status']

X_gender = train_data[text_feature]
Y_gender = train_data.gender


SGDModel_gender = SGDClassifier(shuffle = True)
count_vect=CountVectorizer()
Tf_Idf_Transformer = TfidfTransformer()
accuracy = 0
gender_Kfold = KFold(n_splits= 10, shuffle = True)
for training_index, test_index in gender_Kfold.split(X_gender, Y_gender) :

    X_train_gender, X_test_gender =  X_gender.loc[training_index,], X_gender.loc[test_index,]
    Y_train_gender, Y_test_gender =  Y_gender.loc[training_index,], Y_gender.loc[test_index,]

    gender_training_count_vect = count_vect.fit_transform(X_train_gender.Status)
    gender_training_Tf_Idf_Transformer = Tf_Idf_Transformer.fit_transform(gender_training_count_vect)

    SGDModel_gender.fit(gender_training_Tf_Idf_Transformer,Y_train_gender)

    gender_test_count_vect = count_vect.transform(X_test_gender.Status)
    gender_test_Tf_Idf_Transformer = Tf_Idf_Transformer.transform(gender_test_count_vect)

    gender_predicted = SGDModel_gender.predict(gender_test_Tf_Idf_Transformer)
    accuracy+=accuracy_score(Y_test_gender,gender_predicted)
gender_test_count_vect = count_vect.transform(test_data.Status)
gender_test_Tf_Idf_Transformer = Tf_Idf_Transformer.transform(gender_test_count_vect)

gender_predicted = SGDModel_gender.predict(gender_test_Tf_Idf_Transformer)
y_predicted = np.int_(gender_predicted)


userId_to_GenderByStatus = dict()
for index, row in test_data.iterrows():
    userid = getattr(row , 'userid')
    #print(userid)
    #print(y_predicted[index])
    userId_to_GenderByStatus[userid] = y_predicted[index]

###############################################Predicting_Age_from_Status##################################################
print("Predicting Age from Status...")

text_feature = ['Status']
X_age = train_data[text_feature]
Y_age = train_data.age_grp
#print(Y_age)
#LogModel_age = linear_model.LogisticRegression()
SGDModel_age = SGDClassifier(shuffle = True)
count_vect_age=CountVectorizer()
Tf_Idf_Transformer_age = TfidfTransformer()
accuracy = 0
age_Kfold = KFold(n_splits= 10, shuffle = True)
for training_index, test_index in age_Kfold.split(X_age, Y_age) :

    X_train_age, X_test_age =  X_age.loc[training_index,], X_age.loc[test_index,]
    Y_train_age, Y_test_age =  Y_age.loc[training_index,], Y_age.loc[test_index,]

    age_training_count_vect = count_vect_age.fit_transform(X_train_age.Status)
    age_training_Tf_Idf_Transformer = Tf_Idf_Transformer_age.fit_transform(age_training_count_vect)

    SGDModel_age.fit(age_training_Tf_Idf_Transformer,Y_train_age)

    age_test_count_vect = count_vect_age.transform(X_test_age.Status)
    age_test_Tf_Idf_Transformer = Tf_Idf_Transformer_age.transform(age_test_count_vect)

    age_predicted = SGDModel_age.predict(age_test_Tf_Idf_Transformer)
    accuracy+=accuracy_score(Y_test_age,age_predicted)
age_test_count_vect = count_vect_age.transform(test_data.Status)
age_test_Tf_Idf_Transformer = Tf_Idf_Transformer_age.transform(age_test_count_vect)

age_predicted = SGDModel_age.predict(age_test_Tf_Idf_Transformer)
#y_predicted = np.string_(age_predicted)
y_predicted = np.string_(age_predicted)

############################################################################################################

userId_to_AgeByStatus = dict()
for index, row in test_data.iterrows():
    userid = getattr(row , 'userid')
    #print(userid)
    #print(age_predicted[index])
    userId_to_AgeByStatus[userid] = age_predicted[index]

###################### Use this when test data doesn't contain labels####################################
print("Saving output")
for row in testdf.loc[:, ['userid']].iterrows():
    userid = getattr(row[1], 'userid')
    gender_status = userId_to_GenderByStatus[userid]
    final_gender=gender_status
    #print(userid)
    #print(final_gender)
    age_status = userId_to_AgeByStatus[userid]
    age_grp = age_status
    #print(age_grp)

#for row in testdf_profile_emotions.loc[:, ['userid']].iterrows():
#    userid = getattr(row[1], 'userid')
    ope = userid_to_emotion_ope_dictionary[userid]
    con = userid_to_emotion_con_dictionary[userid]
    ext = userid_to_emotion_ext_dictionary[userid]
    agr = userid_to_emotion_agr_dictionary[userid]
    neu = userid_to_emotion_neu_dictionary[userid]


    xml = '<user id="{0}"\nage_group="{1}"\ngender="{2}"\nextrovert="{5}"\nneurotic="{7}"\nagreeable="{6}"\nconscientious="{4}"\nopen="{3}"\n/>'.format(
        userid, age_grp, final_gender, ope, con, ext, agr, neu)
    text_file = open('{0}/{1}.xml'.format(output_directory, userid), "w")


    text_file.write(xml)
    text_file.close()
