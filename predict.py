# Code for 'predict.py' file.
# You have already created this ML model in one of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

@st.cache()
def prediction(model, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
               BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):
    X = cc_df.iloc[:, :-1] 
    y = cc_df['Default']
    scaler= StandardScaler()
    X= scaler.fit_transform(X)
    
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=42)
    X_train,y_train= SMOTE.fit_resample(X_train,y_train)
    
    # RFC Model
    rf= RandomForestClassifier()
    rf.fit(X_train,y_train)
    score = rf.score(X_train, y_train)
    # pred_rf= rf.predict(X_test)
    default = model.predict([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
               BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])
    default = default[0]
    
    if default == 1:
        return "the client is a defaulter".upper()
    elif glass_type == 0:
        return "the client is not a defaulter".upper()

#     print("Random Forest Accuracy is:", accuracy_score(y_test, pred_rf))

#     print(classification_report(y_test,pred_rf ))

#     plot_confusion_matrix(rf, X_test, y_test, cmap="Blues_r")

    return default

# Defining a function 'app()' which accepts 'car_df' as an input.
def app(cc_df): 
    st.markdown("<p style='color:blue;font-size:25px'>Credit Card Defaulter Classifier", unsafe_allow_html = True) 
    classifier = st.selectbox("Classifier", 
                                 ('Random Forest Classifier', 'XGBoost Classifier'))
    
    if classifier == 'Random Forest Classifier':
        st.subheader("Model Hyperparameters")
        n_estimators_input = st.number_input("Number of trees in the forest", 100, 5000, step = 10)
        max_depth_input = st.number_input("Maximum depth of the tree", 1, 100, step = 1)
        
        if st.sidebar.button('Classify'):
            st.subheader("Random Forest Classifier")
            rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
            rf_clf.fit(X_train,y_train)
            accuracy = rf_clf.score(X_test, y_test)
            default = prediction(rf_clf, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
               BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6)
           
            st.write(default)
            st.write("Accuracy", accuracy.round(2))
            plot_confusion_matrix(rf_clf, X_test, y_test)
            st.pyplot()
            
    if classifier == 'XGBoost Classifier':
        st.subheader("Model Hyperparameters")
        max_depth_ = st.number_input("Maximum Depth", 6, 20, step = 2)
        
        if st.button('Classify'):
            st.subheader("XGBoost Classifier")
            xgboost = xgb.XGBClassifier(max_depth = max_depth_)
            xgboost.fit(X_train,y_train)
            accuracy = xgboost.score(X_test, y_test)
            default = prediction(xgboost, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3,
               BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6)
           
            st.write(default)
            st.write("Accuracy", accuracy.round(2))
            plot_confusion_matrix(xgboost, X_test, y_test)
            st.pyplot()        


    st.subheader("Select Values:")
    LIMIT_BAL = st.slider("Limit Balance", float(cc_df["LIMIT_BAL"].min()), float(cc_df["LIMIT_BAL"].max())) 
    SEX = st.radio("Gender", ("Male", "Female"))
    if SEX == 'Male':
        SEX = 1
    else:
        SEX = 2
    EDUCATION = st.radio("EDUCATION", ("School Graduate", "University Graduate", "High School Graduate", "Graduate from Unknown Institution", "Unknown Education Status"))
    if EDUCATION == 'School Graduate':
        EDUCATION = 1
    elif EDUCATION == 'University Graduate':
        EDUCATION = 2
    elif EDUCATION == 'High School Graduate':
        EDUCATION = 3
    elif EDUCATION == 'Graduate from Unknown Institution':
        EDUCATION = 4
    elif EDUCATION == 'Unknown Education':
        EDUCATION = 5
    eng_siz = st.slider("Engine Size", int(car_df["enginesize"].min()), int(car_df["enginesize"].max()))
    hor_pow = st.slider("Horse Power", int(car_df["horsepower"].min()), int(car_df["horsepower"].max()))    
    drw_fwd = st.radio("Is it a forward drive wheel car?", ("Yes", "No"))
    
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

