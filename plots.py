# Code for 'plots.py' file.
# Import necessary modules 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function 'app()' which accepts 'car_df' as an input.
def app(cc_df):
    st.header('Visualise data')
    # Remove deprecation warning.
    st.set_option('deprecation.showPyplotGlobalUse', False)

#     # Subheader for scatter plot.
#     st.subheader("Scatter plot")
#     # Choosing x-axis values for scatter plots.
#     features_list = st.multiselect("Select the x-axis values:", 
#                                             ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
#     # Create scatter plots.
#     for feature in features_list:
#         st.subheader(f"Scatter plot between {feature} and price")
#         plt.figure(figsize = (12, 6))
#         sns.scatterplot(x = feature, y = 'price', data = car_df)
#         st.pyplot()

    # Add a multiselect widget to allow the user to select multiple visualisation.
    # Add a subheader in the sidebar with label "Visualisation Selector"
    st.subheader("Visualisation Selector")

    # Add a multiselect in the sidebar with label 'Select the charts or plots:'
    # and pass the remaining 3 plot types as a tuple i.e. ('Box Plot', 'Correlation Heatmap').
    # Store the current value of this widget in a variable 'plot_types'.
    plot_types = st.multiselect("Select charts or plots:", ('Count Plot', 'Correlation Heatmap'))
   

    # Create box plot using the 'seaborn' module and the 'st.pyplot()' function.
    if 'Count Plot' in plot_types:
        st.subheader("Count Plot")
        columns = st.selectbox("Select the column to create its box plot",
                                      ('EDUCATION', 'MARRIAGE', 'SEX'))
        plt.figure(figsize = (12, 2))
        plt.title(f"Box plot for {columns}")
        ax=sns.countplot(data=cc_df, x=columns, hue="Default")
        for label in ax.containers:
            ax.bar_label(label)
        plt.xticks([0,1], labels=["male", "female"])
        plt.title("Defaulters based on Gender")
        plt.show()
        #sns.countplot(cc_df[columns], hue=)
        st.pyplot()

    # Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
    if 'Correlation Heatmap' in plot_types:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize = (8, 5))
        ax = sns.heatmap(cc_df.corr(), annot = True) # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim() # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5) # Increasing the bottom and decreasing the bottom margins respectively.
        st.pyplot()
