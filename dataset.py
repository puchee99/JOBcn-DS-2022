from __future__ import annotations
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch 
from torch.autograd import Variable


#------------------------collect .csv file and load---------------------------------
def get_path(filename: str = 'train.csv') -> tuple[str, str]:
    curr_path = str(os.getcwd())
    path_data = curr_path + '/data/' + filename
    file_extension = path_data.split(".")[-1]
    return path_data, file_extension

def get_dataframe(path_data: str, file_extension: str) -> pd.DataFrame:
    if file_extension == 'xlsx':
        df = pd.read_excel(path_data, engine='openpyxl')
    elif file_extension == 'xls':
        df = pd.read_excel(path_data)
    elif file_extension == 'csv':
        df = pd.read_csv(path_data)
    return df 

#------------------------get features to train, target_values, name_cols---------------------------------
def clean_data(df:pd.DataFrame, label_col_name:str, select_corr_col:bool=True, resample_df:bool=True) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    df_train = df.copy()
    if resample_df:
        df_0 = df_train[df_train[label_col_name] == 0]
        df_1 = df_train[df_train[label_col_name] == 1]
        df_2 = df_train[df_train[label_col_name] == 2]

        new_0 = resample(df_0, replace=True, n_samples=1, random_state=123) 
        new_1 = resample(df_1, replace=True, n_samples=1, random_state=123)
        new_2 =  resample(df_2, replace=True, n_samples=1000, random_state=123)

        df_train =pd.concat([df_train, new_0, new_1, new_2])
    y = df_train[label_col_name]
    if select_corr_col:
        #grab the variables that are related to the target column
        corr = df_train.corr().abs()[label_col_name]
        corr = corr[corr != 1]
        corr = corr[corr > 0.05]
        select_columns = list(corr.index)
        df_select_columns = df_train[select_columns]
        df_train = df_select_columns.copy()

    return df_train, y, select_columns

def create_datasets(filename:str, target_col:str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    #collect data from filename and clean it
    path, extension = get_path(filename)
    df = get_dataframe(path, extension)
    return clean_data(df, label_col_name=target_col )

#--------------------------- get model data ------------------------------------------------
def get_train_test_val(X,y) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    #split train, test, validation
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
    
    #standardize the variables and transform the X values into an array
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_val, y_val = np.array(X_val), np.array(y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_train_test_val_variable(X_train, X_test, X_val, y_train, y_test, y_val) -> tuple[Variable, Variable, Variable, Variable, Variable, Variable]:
    #transform array to pythorch Variables
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()
    X_val  = Variable(torch.from_numpy(X_val)).float()
    y_val  = Variable(torch.from_numpy(y_val)).long()
    return X_train, X_test, X_val, y_train, y_test, y_val

#--------------------------------------aux---------------------------------------
def create_dataset_without_true_label(train_file:str, test_file:str,label_name:str, index_col_name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path, extension = get_path(train_file)
    df = get_dataframe(path, extension)
    _,_, selected_col = clean_data(df, label_col_name=label_name )

    path_test, extension_test = get_path(test_file)
    df_test = get_dataframe(path_test, extension_test)
    return df_test[selected_col], df_test[[index_col_name]]
    
def get_test_without_label_variable(df: pd.DataFrame):
    X = np.asarray(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return Variable(torch.from_numpy(X_scaled)).float()

def get_class_distribution(y: np.array) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
