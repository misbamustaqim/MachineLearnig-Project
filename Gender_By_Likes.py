
def Get_LikeId_to_Gender_Dictionary_From_Relation(df_likes, UserId_To_Gender):
    LikeIdGender = dict()
    LikeID_to_Weight = dict()
    male_likes = 0
    female_likes = 0
    for row in df_likes.iterrows():
        userid = row[1]['userid']
        likeId = row[1]['like_id']

        gender = UserId_To_Gender.get(userid)
        if gender == 0:
            if likeId not in LikeIdGender:
                LikeIdGender[likeId] = [0, 0]
            LikeIdGender[likeId][0] = LikeIdGender[likeId][0] + 1
            male_likes = male_likes + 1

        else:
            if likeId not in LikeIdGender:
                LikeIdGender[likeId] = [0, 0]
            LikeIdGender[likeId][1] = LikeIdGender[likeId][1] + 1
            female_likes = female_likes + 1

    for key, value in LikeIdGender.items():
        Difference = LikeIdGender[key][0] - LikeIdGender[key][1]
        Sum = LikeIdGender[key][0] + LikeIdGender[key][1]
        square = Difference ** (2)
        weightOfPage = square / Sum
        if (Difference > 0):
            LikeID_to_Weight[key] = (1 / male_likes)
        else:
            if (weightOfPage <= 2.0):
                continue
            LikeID_to_Weight[key] = - (1 / female_likes)

    return LikeID_to_Weight


def Predict_Gender_from_TotalWeight(userID_to_totalWeight):
    useridToPredictedGender = dict()
    zero_weight = 0
    predicted_male = 0
    predicted_female = 0
    for key, value in userID_to_totalWeight.items():
        userid = key

        if (value > 0):
            predicted_gender = 0.0
            predicted_male += 1
        else:
            predicted_gender = 1.0
            predicted_female += 1
        useridToPredictedGender[userid] = predicted_gender

        if (value == 0):
            zero_weight = zero_weight + 1

    return useridToPredictedGender


class Gender_By_Likes():
    def run(train_profile_df,df_likes,test_profile_df,df_likes_test):

        print("Predicting gender from likes")
        userID_to_totalWeight = dict()

        def Test_dataset(df_likes_test, LikeID_to_Weight):
            page_not_liked = 0
            for row in df_likes_test.iterrows():
                userid = row[1]['userid']
                likeId = row[1]['like_id']

                if userid not in userID_to_totalWeight:
                    userID_to_totalWeight[userid] = 0

                if (likeId not in LikeID_to_Weight):
                    page_not_liked = page_not_liked + 1
                    # weight = (1.0/male_likes)
                    continue
                else:
                    weight = LikeID_to_Weight[likeId]

                userID_to_totalWeight[userid] = userID_to_totalWeight[userid] + weight
            return userID_to_totalWeight

        UserId_To_Gender = dict(zip(train_profile_df['userid'], train_profile_df['gender']))
        LikeID_to_Weight = Get_LikeId_to_Gender_Dictionary_From_Relation(df_likes, UserId_To_Gender)
        userID_to_totalWeight = Test_dataset(df_likes_test, LikeID_to_Weight)
        return Predict_Gender_from_TotalWeight(userID_to_totalWeight)

