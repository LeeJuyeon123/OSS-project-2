import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

def sort_dataset(dataset_df):
    p1 = dataset_df.sort_values(by='year', ascending=True)
    return p1


def split_dataset(dataset_df):
    data=dataset_df.drop(columns=['salary'])
    target = dataset_df[['salary', 'war']]
    target=target.mul(0.001)
    target=target['salary']
    target=target.astype(np.int32)
    x_train, x_test, y_train, y_test= train_test_split(data, target,test_size=0.1015, random_state=0, shuffle=False) #test_size를 0.1015로 하여 1718행을 기준으로 나뉘어지도록 했습니다.
    return x_train, x_test, y_train, y_test

def extract_numerical_cols(dataset_df):
    data=dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    return data




def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_cls=DecisionTreeClassifier()
    dt_cls.fit(X_train, Y_train)
    predicted=dt_cls.predict(X_test)
    return predicted


def train_predict_random_forest(X_train, Y_train, X_test):
    rf_cls=RandomForestClassifier()
    rf_cls.fit(X_train, Y_train)
    predicted=rf_cls.predict(X_test)
    return predicted



def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe=make_pipeline(
        StandardScaler(),
        SVC()
    )
    svm_pipe.fit(X_train, Y_train)
    predicted=svm_pipe.predict(X_test)
    return predicted



def calculate_RMSE(labels, predictions):
    return np.sqrt(np.mean((predictions-labels)**2))



if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)

    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)


    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)


    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))