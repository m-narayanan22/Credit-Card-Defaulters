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
           
            st.success(default)
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
    MARRIAGE = st.radio("MARRIAGE", ("Married", "Unmarried", "Others"))
    if MARRIAGE == 'Married':
        MARRIAGE = 1
    elif MARRIAGE == 'Unmarried':
        MARRIAGE = 2
    elif MARRIAGE == 'Others':
        MARRIAGE = 3
    AGE = st.slider("Age", int(cc_df["AGE"].min()), float(cc_df["AGE"].max())) 
    PAY_0 = st.slider("Repayment Status in Sep 2005 (-1 denotes no dues)", int(-1, 9)) 
    PAY_2 = st.slider("Repayment Status in Aug 2005 (-1 denotes no dues)", int(-1, 9))
    PAY_3 = st.slider("Repayment Status in July 2005 (-1 denotes no dues)", int(-1, 9))
    PAY_4 = st.slider("Repayment Status in June 2005 (-1 denotes no dues)", int(-1, 9))
    PAY_5 = st.slider("Repayment Status in May 2005 (-1 denotes no dues)", int(-1, 9))
    PAY_6 = st.slider("Repayment Status in Apr 2005 (-1 denotes no dues)", int(-1, 9))
    BILL_AMT1 = st.slider("Bill Amount in Sep 2005", float(cc_df["BILL_AMT1"].min()), float(cc_df["BILL_AMT1"].max()))
    BILL_AMT2 = st.slider("Bill Amount in Aug 2005", float(cc_df["BILL_AMT2"].min()), float(cc_df["BILL_AMT2"].max()))
    BILL_AMT3 = st.slider("Bill Amount in July 2005", float(cc_df["BILL_AMT3"].min()), float(cc_df["BILL_AMT3"].max()))
    BILL_AMT4 = st.slider("Bill Amount in June 2005", float(cc_df["BILL_AMT4"].min()), float(cc_df["BILL_AMT4"].max()))
    BILL_AMT5 = st.slider("Bill Amount in May 2005", float(cc_df["BILL_AMT5"].min()), float(cc_df["BILL_AMT5"].max()))
    BILL_AMT6 = st.slider("Bill Amount in Apr 2005", float(cc_df["BILL_AMT6"].min()), float(cc_df["BILL_AMT6"].max()))
    PAY_AMT1 = st.slider("Previous Payment in Sep 2005", float(cc_df["PAY_AMT1"].min()), float(cc_df["PAY_AMT1"].max()))
    PAY_AMT2 = st.slider("Repayment Status in Aug 2005", float(cc_df["PAY_AMT2"].min()), float(cc_df["PAY_AMT2"].max()))
    PAY_AMT3 = st.slider("Repayment Status in July 2005", float(cc_df["PAY_AMT3"].min()), float(cc_df["PAY_AMT3"].max()))
    PAY_AMT4 = st.slider("Repayment Status in June 2005", float(cc_df["PAY_AMT4"].min()), float(cc_df["PAY_AMT4"].max()))
    PAY_AMT5 = st.slider("Repayment Status in May 2005", float(cc_df["PAY_AMT5"].min()), float(cc_df["PAY_AMT5"].max()))
    PAY_AMT6 = st.slider("Repayment Status in Apr 2005", float(cc_df["PAY_AMT6"].min()), float(cc_df["PAY_AMT6"].max()))
   
    
    # When 'Predict' button is clicked, the 'prediction()' function must be called 
    # and the value returned by it must be stored in a variable, say 'price'. 
    # Print the value of 'price' and 'score' variable using the 'st.success()' and 'st.info()' functions respectively.
#     if st.button("Classify"):
#         st.subheader("Classification result:")
#         price, score, car_r2, car_mae, car_msle, car_rmse = prediction(car_df, car_wid, eng_siz, hor_pow, drw_fwd, com_bui)
#         st.success("The predicted price of the car: ${:,}".format(int(price)))
#         #st.info("Accuracy score of this model is: {:2.2%}".format(score))
#         st.info(f"R-squared score of this model is: {car_r2:.3f}")  
#         st.info(f"Mean absolute error of this model is: {car_mae:.3f}")  
#         st.info(f"Mean squared log error of this model is: {car_msle:.3f}")  
#         st.info(f"Root mean squared error of this model is: {car_rmse:.3f}")

