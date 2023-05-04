# Code for 'predict.py' file.
# You have already created this ML model in one of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

@st.cache()
def prediction(cc_df, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
               BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):
    X = cc_df.iloc[:, :-1] 
    y = cc_df['Default']
    scaler= StandardScaler()
    X= scaler.fit_transform(X)
    
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=42)
    X_train,y_train= SMOTE.fit_resample(X_train,y_train)
    
    
    rf= RandomForestClassifier()

    rf.fit(X_train,y_train)

    pred_rf= rf.predict(X_test)

#     print("Random Forest Accuracy is:", accuracy_score(y_test, pred_rf))

#     print(classification_report(y_test,pred_rf ))

#     plot_confusion_matrix(rf, X_test, y_test, cmap="Blues_r")


    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    score = lin_reg.score(X_train, y_train)

    price = lin_reg.predict([[car_width, engine_size, horse_power, drive_wheel_fwd, car_comp_buick]])
    price = price[0]

    y_test_pred = lin_reg.predict(X_test)
    test_r2_score = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_msle = mean_squared_log_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return price, score, test_r2_score, test_mae, test_msle, test_rmse

# Defining a function 'app()' which accepts 'car_df' as an input.
def app(car_df): 
    st.markdown("<p style='color:blue;font-size:25px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    car_wid = st.slider("Car Width", float(car_df["carwidth"].min()), float(car_df["carwidth"].max()))     
    eng_siz = st.slider("Engine Size", int(car_df["enginesize"].min()), int(car_df["enginesize"].max()))
    hor_pow = st.slider("Horse Power", int(car_df["horsepower"].min()), int(car_df["horsepower"].max()))    
    drw_fwd = st.radio("Is it a forward drive wheel car?", ("Yes", "No"))
    if drw_fwd == 'No':
        drw_fwd = 0
    else:
        drw_fwd = 1
    com_bui = st.radio("Is the car manufactured by Buick?", ("Yes", "No"))
    if com_bui == 'No':
        com_bui = 0
    else:
        com_bui = 1
    
    # When 'Predict' button is clicked, the 'prediction()' function must be called 
    # and the value returned by it must be stored in a variable, say 'price'. 
    # Print the value of 'price' and 'score' variable using the 'st.success()' and 'st.info()' functions respectively.
    if st.button("Predict"):
        st.subheader("Prediction results:")
        price, score, car_r2, car_mae, car_msle, car_rmse = prediction(car_df, car_wid, eng_siz, hor_pow, drw_fwd, com_bui)
        st.success("The predicted price of the car: ${:,}".format(int(price)))
        #st.info("Accuracy score of this model is: {:2.2%}".format(score))
        st.info(f"R-squared score of this model is: {car_r2:.3f}")  
        st.info(f"Mean absolute error of this model is: {car_mae:.3f}")  
        st.info(f"Mean squared log error of this model is: {car_msle:.3f}")  
        st.info(f"Root mean squared error of this model is: {car_rmse:.3f}")

