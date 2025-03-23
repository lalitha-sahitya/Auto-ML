import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def display():
    st.title("📊 Data Visualization")

    if "dataset" not in st.session_state:
        st.error(" No dataset found! Please upload a CSV first.")
        return

    df = st.session_state.dataset 
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    column = st.selectbox("Choose a column to visualize:", df.columns)

    plot_type = st.radio("Select Plot Type", ["Histogram", "Boxplot", "Scatterplot", "Correlation Heatmap"])

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

    elif plot_type == "Scatterplot":
        col2 = st.selectbox("📌 Choose second column:", df.columns)
        st.write(f"### Scatterplot: {column} vs {col2}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[column], y=df[col2], ax=ax)
        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
