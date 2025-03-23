import streamlit as st
import requests
import pandas as pd
from regression.frontend import visualization

BACKEND_URL = "http://127.0.0.1:5002"

import logging
logging.basicConfig(level=logging.DEBUG)

def display():
    st.title("Regression Model Trainer")
    st.write("Train and evaluate multiple regression models.")

    if "show_visualization" not in st.session_state:
        st.session_state.show_visualization = False  
    if "dataset" not in st.session_state:
        st.session_state.dataset = None  

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df 
            st.success("File uploaded successfully!")
            st.write("### Dataset Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        with st.spinner("Training multiple models..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{BACKEND_URL}/train", files=files)

            if response.status_code == 200:
                result = response.json()
                best_model_r = result["best_model"]
                model_results = result["results"]

                st.success(f"**Best Performing Model: {best_model_r}**")
                st.subheader("Model Performance Comparison")
                performance_df = pd.DataFrame([
                    {"Model": model, "MSE": details["Mean Squared Error"], "R²": details["R² Score"]}
                    for model, details in model_results.items()
                ])
                st.dataframe(performance_df, use_container_width=True)

                model_response = requests.get(f"{BACKEND_URL}/download_model")
                if model_response.status_code == 200:
                    with open("best_model.pkl", "wb") as f:
                        f.write(model_response.content)

                    with open("best_model.pkl", "rb") as f:
                        st.download_button(
                            label="Download Best Model",
                            data=f,
                            file_name="best_model.pkl",
                            mime="application/octet-stream"
                        )
                else:
                    st.error("Model not found! Please train a model first.")

        if st.button("Visualize Data"):
            st.session_state.show_visualization = True
            st.rerun()
    if st.session_state.show_visualization and st.session_state.dataset is not None:
        visualization.display()
    else:
        st.write("Please upload a CSV file to train your model or download from below datasets.")
    st.markdown("---")
    st.markdown("## Example Regression Datasets ")
    example_datasets = {
        "Wine Quality": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "Diabetes": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "Boston Housing": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "California": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
        "Auto MPG": "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
        "Concrete Strength": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "Energy Efficiency": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "Bike Sharing Demand": "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv",
        "Student Performance": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "Weather Data": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
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


if __name__ == "__main__":
    display()
