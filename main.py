import pandas as pd

#Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.svm import SVR 

#Model selection
from sklearn.model_selection import train_test_split

#Evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification

#Machine Learning Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def data_ingest(link):

    """
    This function takes in an url 
    and outputs the dataframe that is stored in the url.
    """

    dataframe = pd.read_csv(link)
    return dataframe


def process_data(dataframe):

    """
    This function takes in a dataframe 
    and removes all duplicate rows.
    It also changes 'weather' values to lowercase to prevent extra unique values.
    It also converts negative user values to 0 and prints out the unique values.
    """

    #Removing duplicate rows
    dataframe = dataframe.drop_duplicates()
    dataframe.shape

    #Converting values in 'weather' column to all lowercase
    dataframe['weather'] = dataframe['weather'].str.lower()
    print(dataframe.weather.unique())

    #Replacing 'weather' values with missing letters (letter 'c')
    dataframe['weather'] = dataframe['weather'].replace('lear','clear')
    dataframe['weather'] = dataframe['weather'].replace('loudy','cloudy')
    print(dataframe['weather'].unique())

    #Converting negative values in the 'user' columns to positive
    dataframe['guest-users'][dataframe['guest-users']<0] = 0
    dataframe['registered-users'][dataframe['registered-users']<0] = 0
    # to check whether there are values that are still lesser than 0
    print((dataframe['guest-users']<0).unique()) 
    print((dataframe['registered-users']<0).unique()) 

    return dataframe

def feature_engineering(dataframe):

    """
    This function creates the attribute 'total-users' from 'guest-users' and 'registered-users' attributes.
    It also creates 'year', 'month' and 'weekday_name' from 'date' attribute.
    This function returns the dataframe with all the newly created variables in it.

    """


    #Attribute creation for 'total users', 'days' and 'months'
    dataframe['total-users'] = dataframe['guest-users'] + dataframe['registered-users']

    dataframe['year'] = pd.DatetimeIndex(dataframe['date']).year
    dataframe['month'] = pd.DatetimeIndex(dataframe['date']).month
    dataframe['weekday_name'] = pd.DatetimeIndex(dataframe['date']).weekday_name


    return dataframe

def feature_engineering2(dataframe):

    """
    1) This function converts all categorical values ('weather' and 'weekday_name')
    into numerical values instead.
    2)It then drops all unnecessary columns that are required for testing the model.
    3) Finally, it returns a dataframe that is cleaned and ready to feed into machine
    learning models for testing.
    """
    #Converting categorical attributes to numerical attributes
    weather_dummy = pd.get_dummies(dataframe['weather'])
    dataframe2 = pd.concat([dataframe,weather_dummy],1)
    dataframe2.head()

    day_dummy = pd.get_dummies(dataframe2['weekday_name'])
    dataframe2 = pd.concat([dataframe2,day_dummy],1)
    dataframe2.head()

    #Drop unncessary columns
    dataframe2.drop(['date','weekday_name','weather','guest-users','registered-users'], axis = 1, inplace = True)  

    return dataframe2

def reorder_columns(dataframe2):

    """
    This function basically reorders the columns of a dataframe (for visual purpose only)
    before feeding it to machine learning models for testing

    """
    #Changing the orders of the columns to have time values at the left
    new_order = ['year','month','hr','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday',\
                 'Sunday','clear','cloudy','light snow/rain',\
                 'heavy snow/rain','temperature','feels-like-temperature','relative-humidity','windspeed','psi',\
                 'total-users']
    dataframe2 = dataframe2[new_order]


    return dataframe2

def split_data_to_train_test(dataframe2):

    """
    This function takes in a dataset and splits it into
    test and train sets in the ratio of 30% to 70%.
    """

    x_train,x_test,y_train,y_test=train_test_split(dataframe2.drop('total-users',axis = 1),dataframe2['total-users'],test_size=0.3,random_state=2)


    return x_train,x_test,y_train,y_test



def train_fit_evaluate(x_train,x_test,y_train,y_test):

    """
    This function takes in dataframe, x_train,x_test,y_train and y_test.
    It will then train a variety of machine learning models.
    It will finally output the accuracy of each of the machine learning models in predicting y_test.
    """
    # Initialize pipeline for the models
    pl_rfr = Pipeline(steps=[('scl',StandardScaler()),('rfr',RandomForestRegressor(n_estimators=100))], verbose=True)
    pl_abr = Pipeline(steps=[('scl',StandardScaler()),('abr',AdaBoostRegressor())], verbose=True)
    pl_br = Pipeline(steps=[('scl',StandardScaler()),('br',BaggingRegressor())], verbose=True)
    pl_svr = Pipeline(steps=[('scl',StandardScaler()),('svr',SVR(gamma='scale',kernel = 'rbf'))], verbose=True)
    pl_lr = Pipeline(steps=[('scl',StandardScaler()),('lr',LinearRegression())], verbose=True)
    
    #List of pipelines
    pipelines = [pl_rfr, pl_abr, pl_br, pl_svr, pl_lr]

    #Fit the pipelines that are in the list above
    for pipe in pipelines:
        pipe.fit(x_train, y_train)

    #Dictionary of pipelines and classifier types for ease of reference
    pipe_dict = {0: 'Random Forest Regressor', 1: 'Ada Boost Regressor', 2: 'Bagging Regressor', 3: 'SVR'\
             , 4: 'Linear Regression' }

    #Print accuracy for pipelines   
    for i,model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(x_test,y_test)))


    #Define parameters for for loop below
    best_accuracy=0.0
    best_classifier=0
    best_pipeline=""

    #Comparing models to come up with best models   
    for i,model in enumerate(pipelines):
        if model.score(x_test,y_test)>best_accuracy:
            best_accuracy=model.score(x_test,y_test)
            best_pipeline=model
            best_classifier=i
    print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))
    



    return

def main():
 
    if __name__ == "__main__":
        main()





