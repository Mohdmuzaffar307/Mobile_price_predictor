import pandas as pd
from jinja2.utils import missing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import streamlit as st
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from xgboost import XGBRegressor

from sklearn.feature_selection import SelectKBest
import joblib

df=pd.read_csv('smartphones_cleaned_v6.csv')
df=df[['brand_name','has_5g','processor_brand','battery_capacity','ram_capacity',"internal_memory",'price']]
x=df.drop(columns='price')
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

transformer=ColumnTransformer(
    [
        ('imputer',SimpleImputer(strategy='median'),['battery_capacity']),
        ('ohe',OneHotEncoder(sparse_output=False,drop='first'),['brand_name','processor_brand']),
        ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),['has_5g'])

        
    ],remainder='passthrough'
)
# transformer.set_output(transform='pandas')
# x_train=transformer.fit(x_train)
# x_test=transformer.transform(x_test)
scale=RobustScaler()

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
# clf=RandomForestRegressor(random_state=42)
# clf=XGBRegressor(random_state=42,nthread=-1)
# clf=DecisionTreeRegressor(random_state=42)
clf=BaggingRegressor(estimator=XGBRegressor(random_state=42),n_estimators=20,max_samples=0.30,bootstrap=True,random_state=42)



# cross_val_score(clf,x_train,y_train,cv=5,scoring='r2').mean()
pipe=Pipeline([
    ("pre",transformer),
    ("scale",scale),
    ('model',clf)
])

pipe.fit(x_train,y_train)
joblib.dump(pipe,open('pipe1.joblib','wb'))





