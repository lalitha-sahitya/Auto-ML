import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def is_regression_dataset(df):
    """Checks if the dataset is suitable for regression by verifying the target column is continuous."""
    if df.select_dtypes(include=['number']).shape[1] < 2:
        return False  # Need at least two numerical columns for regression analysis
    return True
def display():
    st.title("📊 Regression Data Visualization")

    if "dataset" not in st.session_state:
        st.error("No dataset found! Please upload a CSV first.")
        return

    df = st.session_state.dataset  
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    plot_type = st.radio("Select Plot Type", [
        "Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap",
        "Pairplot", "Regression Plot", "Residual Plot"
    ])

    if plot_type in ["Histogram", "Boxplot"]:
        column = st.selectbox("Choose a column:", df.columns, key="single_column")

        if plot_type == "Histogram":
            st.write(f"### Histogram of {column}")
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Boxplot":
            st.write(f"### Boxplot of {column}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)

    elif plot_type in ["Scatterplot", "Regression Plot", "Residual Plot"]:
        col1, col2 = st.columns(2) 
        with col1:
            x_axis = st.selectbox("Choose X-axis:", df.columns, key=f"{plot_type}_x")
        with col2:
            y_axis = st.selectbox("Choose Y-axis:", df.columns, key=f"{plot_type}_y")

        st.write(f"### {plot_type}: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()

        if plot_type == "Scatterplot":
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == "Regression Plot":
            sns.regplot(x=df[x_axis], y=df[y_axis], ax=ax)
        elif plot_type == "Residual Plot":
            sns.residplot(x=df[x_axis], y=df[y_axis], ax=ax)

        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif plot_type == "Pairplot":
        st.write("### Pairplot of Numeric Columns")
        fig = sns.pairplot(df)
        st.pyplot(fig)
