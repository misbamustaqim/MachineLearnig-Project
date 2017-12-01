# MachineLearnig-Project

## Files

* tcss555: It is a script to run the prediction algorithm and save the results in output file.
Input it should be executed in following manner:
tcss555 -i /data/public-test-data -o ~/output/

* project2.py: Is the main python file which reads the training and test data and refers different files to run prediction algorithm. It also saves the output

* Gender_By_Status.py: This file predicts the gender of the user using the status messages. It uses naive bayes classifier

* Gender_By_Likes.py: This file predicts the gender of the users using the likes of the user.

* Age_by_Likes.py: This file predicts the age of users based on the likes

* personalities_by_Likes.py: This file predicts the emotions of users based on the likes

* testAccuracy.py: similar to project2.py, this file gets the input, splits it into test data and train data and calls the other algorithms to get the prediction. But instead of saving the output, it calculates the accuracy of the result. It is convenient to use this file to calculate accuracy rather than modifying project2.py

