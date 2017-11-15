def Get_LikeId_to_age_Dictionary_From_Relation(df_likes, Source_to_age, source, destination):
    LikeID_to_Age_dictionary = dict()
    for row in df_likes.iterrows():
        source_id = row[1][source]
        destination_id = row[1][destination]
        if source_id not in Source_to_age:
            continue
        else:
            Emotion_array = Source_to_age[source_id]
            if destination_id not in LikeID_to_Age_dictionary:
                LikeID_to_Age_dictionary[destination_id] = [0,0,0,0]
            dict_value = LikeID_to_Age_dictionary[destination_id]
            #print(dict_value)

            if(destination_id not in Source_to_age):
                continue

            if (Source_to_age[destination_id] >= 00 and Source_to_age[destination_id] < 25):
                dict_value[0] += 1
            if (Source_to_age[destination_id] >= 25 and Source_to_age[destination_id] < 35):
                dict_value[1] += 1
            if (Source_to_age[destination_id] >= 35 and Source_to_age[destination_id] < 50):
                dict_value[2] += 1
            if (Source_to_age[destination_id] >= 50):
                dict_value[3] += 1

    for key, value in LikeID_to_Age_dictionary.items():
        age = max(value[0], value[1], value[2], value[3])

        if (age == value[0]):
            age_grp = "xx_24"
        if (age == value[1]):
            age_grp = "25_34"
        if (age == value[2]):
            age_grp = "35_49"
        if (age == value[3]):
            age_grp = "50_xx"

        LikeID_to_Age_dictionary[key] = age_grp

    return LikeID_to_Age_dictionary


def Calculate_Majority_of_age(data_Age):
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

    max_age = max(age_grp_xx_24, age_grp_25_34, age_grp_35_49, age_grp_50_xx)
    if (max_age == age_grp_xx_24):
        age_grp = "xx_24"
    elif (max_age == age_grp_25_34):
        age_grp = "25_34"
    elif (max_age == age_grp_35_49):
        age_grp = "35_49"
    elif (max_age == age_grp_50_xx):
        age_grp = "50_xx"
    return age_grp

class Age_By_Likes():
    print("Predicting Age from likes!!")

    def run(train_profile_df, df_likes, test_profile_df, df_likes_test):
    #--------------------------------------------------------------------------------------------------------------------------------
        def Test_dataset(df_likes_test, LikeId_to_age_Dictionary, mejority_age):
            test_userid_to_age_Dictionary= Get_LikeId_to_age_Dictionary_From_Relation(df_likes_test,LikeId_to_age_Dictionary,'like_id','userid')
            for userid in test_profile_df['userid']:
                if userid not in test_userid_to_age_Dictionary:
                    print("User has no likes")
                    test_userid_to_age_Dictionary[userid] = mejority_age
            return test_userid_to_age_Dictionary
    #------------------------------------------------------------------------------------------------------------------------------------
        data_Age = train_profile_df.loc[:, ['age']]

        UserId_To_Age = dict(zip(train_profile_df['userid'], train_profile_df['age']))
        print(UserId_To_Age)
        LikeId_to_age_Dictionary = Get_LikeId_to_age_Dictionary_From_Relation(df_likes, UserId_To_Age,'userid', 'like_id')
        mejority_age = Calculate_Majority_of_age(data_Age)
        userid_to_age_dictionary = Test_dataset(df_likes_test, LikeId_to_age_Dictionary, mejority_age)
        return userid_to_age_dictionary

        # =================================================
