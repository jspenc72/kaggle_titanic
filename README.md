## kaggle_titanic.ipynb
One of the more basic, open-ended competitions on kaggle. See here for details : https://www.kaggle.com/c/titanic

## Sites which helped most:
* https://www.kaggle.com/cbrogan/xgboost-example-python/code
  * started with this guy's XGBoost (though my end result doesn't resemble it much, thought I should mention it)
* https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
  * I had most of the same ideas on new features and using VotingClassifier, but there were good notes here on really picking features apart and useful graph ideas

## Most success:
* Given Features as-is : Embarked, PClass, Fare, Sex
* Prefix :  binned (by prob of survival)
* Parch and Sibsp : summed and binned
* Ticket Code : gave a slight boost, especially with binning
* Voting Classifier : with 4 or 5 classifiers
  * Each individual classifier seemed to hit a wall at .7999, this put me above
* Tuning to lower bound of accuracy confidence interval (CV mean - 3 CV std's)
  * had major overfitting going on, local accuracy as much as .85, but public accuracy never got above .7999
  * first tuned each classifier to best accuracy
  * then tweaked key params up/down to get a higher LB and/or lower STD
  * this combined with REMOVING features in the "Tried with no success" section put me beyond .80xxx up to .81818

## Tried with no success (or only local success):
* Cabin Code, raw and binning by survival prob groups.
* Cabin Count, raw and binning by surviaval prob groups
* Trimming features with VarianceThreshold
* Naive Bayes and KNN (with scaling on KNN)
* Using family survival rate : Not sure where I saw this mentioned but I think it was a discussion board. Tried taking family names + parch_sibsp as a key.  Marked each as having the majority surviving (or not).  Joined back to train AND test. Seemed like a good idea but a significant number of names from the train set weren't in the test set, so had to mark them unknown.
* Age : I tried multiple ways to make this work, I never saw an increase in accuracy.
* Removing outliers : mostly looking for items over 3 STD's, maybe could have tried something more involved.
* Rarity of prefix : percent with this prefix.



#kaggle_titanic_nn.ipynb
  
This approach starts mostly the same features and uses a homemade neural net as the classifier.  The nn was built from the book 'Make your own Neural Net' (https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork) and I added CV, param grid and some helper functions.  One of the params is the activation function itself but the scaler and internal code (backprop for instance) might need to change to handle activations other than expit.

Current notebook as-is should produce and LB score of 79.45, not better than scikit-learn ensembling approach but it was a worthwhile exercise to learn the basics of neural nets.
