import numpy as np

def Get_LikeId_to_emotion_Dictionary_From_Relation(df_likes, Source_to_emotion, OFFSET, source, destination):
    LikeID_to_Emotion_dictionary = dict()
    for row in df_likes.iterrows():
        source_id = row[1][source]
        destination_id = row[1][destination]
        if source_id not in Source_to_emotion :
            continue
        else:
            Emotion_array = Source_to_emotion[source_id]

            if(source == 'like_id'):
                count = Emotion_array[0]
            else:
                count = 1

            if destination_id not in LikeID_to_Emotion_dictionary:
                LikeID_to_Emotion_dictionary[destination_id] = [0, 0, 0, 0, 0, 0]
            dict_value = LikeID_to_Emotion_dictionary[destination_id]
            dict_value[0] += count
            dict_value[1] += Emotion_array[OFFSET]
            dict_value[2] += Emotion_array[OFFSET+1]
            dict_value[3] += Emotion_array[OFFSET+2]
            dict_value[4] += Emotion_array[OFFSET+3]
            dict_value[5] += Emotion_array[OFFSET+4]
    # for key, value in LikeID_to_Emotion_dictionary.items():
    #     value[1] /= value[0]
    #     value[2] /= value[0]
    #     value[3] /= value[0]
    #     value[4] /= value[0]
    #     value[5] /= value[0]
    return LikeID_to_Emotion_dictionary

def Calculate_total_Average(data_Personality):
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
    Emotion_array= [1, ave_ope, ave_con, ave_ext, ave_agr, ave_neu]
    return Emotion_array


def getRange(WithCount):
    WithCount[1] /= WithCount[0]
    WithCount[2] /= WithCount[0]
    WithCount[3] /= WithCount[0]
    WithCount[4] /= WithCount[0]
    WithCount[5] /= WithCount[0]

    return WithCount


class personalities():

    def run(train_profile_df,df_likes,test_profile_df,df_likes_test):

        print("Predicting emotions from likes!!")

        data_Personality = train_profile_df.loc[:, ['ope', 'con', 'ext', 'agr', 'neu']]

        Userid_to_emotion = train_profile_df.set_index('userid').T.to_dict('list')  # index 3 = ope to 7

        def Test_dataset(df_likes_test, LikeId_to_Average_emotion_Dictionary, average_emotions):
            test_userid_to_Average_emotion_Dictionary= Get_LikeId_to_emotion_Dictionary_From_Relation(df_likes_test,LikeId_to_Average_emotion_Dictionary,1,'like_id','userid')
            for userid in test_profile_df['userid']:
                if userid not in test_userid_to_Average_emotion_Dictionary:
                    #print("User has no likes")
                    test_userid_to_Average_emotion_Dictionary[userid] = average_emotions
                else:
                    test_userid_to_Average_emotion_Dictionary[userid] = getRange(test_userid_to_Average_emotion_Dictionary[userid])
            return test_userid_to_Average_emotion_Dictionary


        LikeId_to_Average_emotion_Dictionary = Get_LikeId_to_emotion_Dictionary_From_Relation(df_likes, Userid_to_emotion,3,'userid','like_id')
        average_emotions = Calculate_total_Average(data_Personality)
        userid_to_emotion_dictionary = Test_dataset(df_likes_test,LikeId_to_Average_emotion_Dictionary, average_emotions)
        return userid_to_emotion_dictionary