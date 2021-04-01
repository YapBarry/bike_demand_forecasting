```python

```

## Forecasting E-Scooter demand

### Pipeline flow would be as per below:
There are 5 parts to our End-to-End Machine Learning Pipeline.
Part 1: Data ingestion - read and save file
Part 2: Process data - get rid of duplicates, lower case etc
Part 3: Feature Engineering - create new features that would be fed into the machine learning models
Part 4: Train, fit model
Part 5: Evaluate model

### Choice of model

In this project, our target variable (total number of users) is a continuous variable.
We are also given several feature variables that will likely affect it.
This looks more like a regression type of Machine Learning project than classification/clustering etc.
Hence the majority of our models tested will be those that are good at tackling regression problems.

We tried a total of 5 models in this project, namely:
1) Linear Regression(baseline) - this acts as a baseline to compare other 4 models against
2) Random Forest Regression
3) Ada Boost Regression
4) Bagging Regression
5) Support Vector Regression (SVR)

From the 5 models above, we choose Random Forest Regression as the model to predict the demand of E-Scooter in the future
as it has by far the highest accuracy.

Random Forest Regressor Test Accuracy: 0.9152849838067368
Ada Boost Regressor Test Accuracy: 0.6019494633880802
Bagging Regressor Test Accuracy: 0.9056501394647972
SVR Test Accuracy: 0.03388304703678946
Linear Regression Test Accuracy: 0.3567665884873823


#### Evaluation of model
We managed to achieve an accuracy of 91.5 (on our test set) for Random Forest Regression.
Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression.
It also runs efficiently on large datasets with many feature variables.
It is almost a 56% improvement over the baseline linear regression of 35.7%.

A close second would be Bagging Regression at 90.6%.
We could adopt Bagging Regression for this problem as well.





