# Code for 'data.py' file
# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st

# Define a function 'app()' which accepts 'cc_df' as an input.
def app(cc_df):
    
    st.header("Credit Card Defaulters Classification App")
    st.text("""
    A credit card is a financial instrument issued by banks with a pre-set credit limit, helping you make cashless transactions.
    Credit card companies make a huge profit by charging about 2% or 3% of the amount as a transaction fee to the merchant (or seller) at the point of sale.
    Credit cards must be used wisely and only when there is a need. From a consumer point of view, despite having risks, credit cards have few advantages too.
    """)
    
    st.header("View Data")
    # Add an expander and display the dataset as a static table within the expander.
    with st.beta_expander("View Dataset"):
        st.table(cc_df)

    st.subheader("Data Description:")
    if st.checkbox("Show summary"):
        st.table(cc_df.describe())

    beta_col1, beta_col2, beta_col3 = st.beta_columns(3)

    # Add a checkbox in the first column. Display the column names of 'cc_df' on the click of checkbox.
    with beta_col1:
        if st.checkbox("Show all column names"):
            st.table(list(cc_df.columns))

    # Add a checkbox in the second column. Display the column data-types of 'cc_df' on the click of checkbox.
    with beta_col2:
        if st.checkbox("Data Information"):
            st.table(cc_df.info)

    # Add a checkbox in the third column followed by a selectbox which accepts the column name whose data needs to be displayed.
    with beta_col3:
        if st.checkbox("View column data"):
            column_data = st.selectbox('Select column', tuple(cc_df.columns))
            st.write(cc_df[column_data])
