import streamlit as st
from PIL import Image
import pandas as pd
import requests
from classification.frontend import classification_ui
from regression.frontend import regression_ui

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.title("AUTO ML - MACHINE LEARNING MODELS TRAINER")
st.sidebar.title("Navigation")

page_options = ["Home", "Classification", "Regression"]
page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.page))

if page:
    st.session_state.page = page 

if st.session_state.page == "Home":

    st.markdown("## Introduction")
    st.write(
        "Machine Learning (ML) is transforming industries by enabling data-driven decision-making. "
        "However, training and selecting the best ML models require expertise in data preprocessing, feature selection, hyperparameter tuning, and model evaluation. "
        "AutoML (Automated Machine Learning) simplifies this process by automating model selection, training, and optimization."
    )
    st.write(
        "Our AutoML - Machine Learning Models Trainer project aims to provide a user-friendly platform "
        "that automates the end-to-end ML workflow, making model building accessible to beginners and experts alike."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Classification"):
            st.session_state.page = "Classification"
            st.rerun() 
    with c2:
        if st.button("Regression"):
            st.session_state.page = "Regression"
            st.rerun()

    st.markdown("---")
    st.markdown("## Example Datasets ")
    example_datasets = {
        "Iris Dataset (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Wine Quality (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "Diabetes (Regression)": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Boston Housing (Regression)": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "California Housing (Regression)": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
        "Auto MPG (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
        "Concrete Strength (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "Energy Efficiency (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "Bike Sharing Demand (Regression)": "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv",
        "Student Performance (Regression)": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "Weather Data (Regression)": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
}

    

    selected_dataset = st.selectbox("Choose a dataset:", list(example_datasets.keys()))

    if selected_dataset:
        dataset_url = example_datasets[selected_dataset]
        df = pd.read_csv(dataset_url)
        st.write(f" Preview of **{selected_dataset}**:")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explore More Datasets"):
                st.write("Visit [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php) for more datasets.")
        with col2:
            st.download_button(
                label="Download Dataset",
                data=df.to_csv(index=False),
                file_name=f"{selected_dataset}.csv",
                mime="text/csv",
            )

elif st.session_state.page == "Classification":
    classification_ui.display()

elif st.session_state.page == "Regression":
    regression_ui.display()

