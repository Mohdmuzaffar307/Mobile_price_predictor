import pandas as pd
# import pickle
import streamlit as st
import numpy as np
import joblib



df=pd.read_csv("smartphones_cleaned_v6.csv",usecols=["brand_name","has_5g","processor_brand","battery_capacity","ram_capacity","internal_memory","price"])
model=joblib.load(open("pipe1.joblib",'rb'))



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

x=pd.DataFrame(np.array([brand,	has_5g	,processor	,battery,ram,internal_memory]).reshape(1,6),columns=df.drop(columns="price").columns)

pointestimate=model.predict(x)[0]
upper_range=pointestimate+1.64*(df['price'].std()/np.sqrt(784))
lower_range=pointestimate-1.64*(df['price'].std()/np.sqrt(784))

btn=st.sidebar.button('Pridict Price')
if btn:
    st.markdown('''
    <h1> The Specification you choose whose price Lies b/w </h1>
    ''',unsafe_allow_html=True)
    st.write(f"{round(lower_range)} --------  {round(upper_range)}")


