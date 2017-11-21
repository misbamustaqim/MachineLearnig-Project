import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn import linear_model
import pandas as pd


class Gender_By_Status():
    def run(train_profile_df, test_profile_df, test_data_directory):

        print("Predicting Gender from status...")

        for index, row in train_profile_df.iterrows():
            data_userid = row['userid']
            userid_to_txt = open('/data/training/text/' + data_userid + '.txt', 'r', errors='ignore')
            data_status = userid_to_txt.read()
            train_profile_df.set_value(index, 'Status', data_status)
            userid_to_txt.close()
        
        for index, row in test_profile_df.iterrows():
            data_userid = row['userid']
            userid_to_txt = open(test_data_directory + '/text/' + data_userid + '.txt', 'r', errors='ignore')
            data_status = userid_to_txt.read()
            test_profile_df.set_value(index, 'Status', data_status)
            userid_to_txt.close()

        data_FBUsers = train_profile_df.loc[:, ['userid', 'age', 'gender','Status']]

        # print(data_FBUsers)

        data_FBUsers_test = test_profile_df.loc[:, ['userid', 'age', 'gender', 'Status']]

        # Getting training and test data
        data_test = data_FBUsers.loc[np.arange(len(data_FBUsers_test)), :]
        data_train = data_FBUsers.loc[np.arange(len(data_FBUsers)), :]

        #Splitting the data into training instances and test instances
        #n = 1500
        #all_Ids = np.arange(len(data_FBUsers))
        #random.shuffle(all_Ids)
        #test_Ids = all_Ids[0:n]
        #train_Ids = all_Ids[n:]
        #data_test = data_FBUsers.loc[test_Ids, :]
        #data_train = data_FBUsers.loc[train_Ids, :]
        count_vect = TfidfVectorizer(stop_words='english',ngram_range=(1,3))

        actual_gender = dict(zip(data_test['userid'], data_test['gender']))
        total = 0
        numFolds = 10
        kf = KFold(len(data_train), numFolds, shuffle=True,random_state=True)
        for train_indices, test_indices in kf:
            #X_train = data_FBUsers.loc[train_indices, :]; y_train = data_FBUsers.loc[train_indices]
            #X_test = data_FBUsers.loc[test_indices, :]; y_test = data_FBUsers.loc[test_indices]
            X_train = count_vect.fit_transform(data_train['Status'])
            y_train = data_train['gender']
            clf = MultinomialNB()
            #clf.fit(X_train, data_train['age_grp'])
            clf.fit(X_train, y_train)
            X_test = count_vect.transform(data_test['Status'])
            y_test = data_test['gender']
        # Testing the Naive Bayes model for Gender prediction
            y_predicted = clf.predict(X_test)
            total += accuracy_score(y_test, y_predicted)
        #X_test = count_vect.transform(data_train['Status'])
        #y_test = data_train['age_grp']
        #y_predicted = clf.predict(X_train)
        accuracy = total / numFolds
        #print("Accuracy_of_Gender: %.2f" % accuracy)
        """
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
	"""
        userId_to_GenderByStatus = dict()
        for index, row in data_FBUsers_test.iterrows():
            userid = getattr(row , 'userid')
            userId_to_GenderByStatus[userid]  = y_predicted[index]

        return userId_to_GenderByStatus
