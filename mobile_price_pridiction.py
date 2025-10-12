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
import pickle
from sklearn.feature_selection import SelectKBest

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

pickle.dump(pipe,open('pipe1.pkl','wb'))
model=pickle.load(open("pipe1.pkl",'rb'))
model.fit(x_train,y_train)


# y_pred=model.predict(x_test)
# from sklearn.metrics import root_mean_squared_error,r2_score
# print(root_mean_squared_error(y_test,y_pred))
# print(r2_score(y_test,y_pred))


brand=st.sidebar.selectbox('Enter brand',list(df['brand_name'].unique()))
has_5g=st.sidebar.selectbox('Choose 5G',[True,False])
processor=st.sidebar.selectbox('Choose Processor',list(df['processor_brand'].unique()))
battery=st.sidebar.selectbox('choose battery capicity',list(df['battery_capacity'].unique()))
ram=st.sidebar.selectbox('choose RAM capicity',list(df['ram_capacity'].unique()))
internal_memory=st.sidebar.selectbox('choose ROM capicity',list(df['internal_memory'].unique()))

# np.array(['sony',True,'snapdragon'	,4000.0	,8.0]).reshape(1,5)

x=pd.DataFrame(np.array([brand,	has_5g	,processor	,battery,ram,internal_memory]).reshape(1,6),columns=x_train.columns)

pointestimate=model.predict(x)[0]
upper_range=pointestimate+1.64*(df['price'].std()/np.sqrt(784))
lower_range=pointestimate-1.64*(df['price'].std()/np.sqrt(784))

btn=st.sidebar.button('Pridict Price')
if btn:
    st.markdown('''
    <h1> The Specification you choose whose price Lies b/w </h1>
    ''',unsafe_allow_html=True)
    st.write(f"{round(lower_range)} --------  {round(upper_range)}")







