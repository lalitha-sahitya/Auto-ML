import streamlit as st
import requests
import pandas as pd
from classification.frontend import visualization 

BACKEND_URL = "http://127.0.0.1:5001"

def display():
    st.title("Classification Model Trainer")
    st.write("Train and evaluate classification models here.")
    if "show_visualization" not in st.session_state:
        st.session_state.show_visualization = False

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataset = df 

        st.write("**File uploaded successfully!**")

        with st.spinner("🚀 Training multiple models..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{BACKEND_URL}/train", files=files)

            if response.status_code == 200:
                result = response.json()
                best_model = result["best_model"]
                model_results = result["results"]

                st.success(f"**Best Model: {best_model}**")
                st.subheader("Model Performance Comparison")
                accuracy_df = pd.DataFrame([
                    {"Model": model, "Accuracy (%)": details["accuracy"]}
                    for model, details in model_results.items()
                ])
                st.table(accuracy_df)

                for model, details in model_results.items():
                    st.subheader(f"🔹 {model}")
                    st.write(f"**Accuracy:** {details['accuracy']}%")
                    st.write("**Classification Report:**")
                    st.dataframe(pd.DataFrame(details["classification_report"]).transpose())

                with st.spinner("Fetching best model..."):
                    model_response = requests.get(f"{BACKEND_URL}/download_model")
                    if model_response.status_code == 200:
                        with open("best_model.pkl", "wb") as f:
                            f.write(model_response.content)

                        with open("best_model.pkl", "rb") as f:
                            st.download_button(
                                label="⬇️ Download Best Model",
                                data=f,
                                file_name="best_model.pkl",
                                mime="application/octet-stream"
                            )
                    else:
                        st.error("Model not found! Please train a model first.")
        if st.button("Visualize Data"):
            st.session_state.show_visualization = True
            st.rerun()

    if st.session_state.show_visualization:
        visualization.display()

    else:
        st.write("Please upload a CSV file to train your model or download from below datasets.")
    st.markdown("---")
    st.markdown("## Example Regression Datasets ")
    example_datasets = {
    "Iris Dataset (Classification)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic (Classification)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Breast Cancer (Classification)": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv",
    "Wine Quality (Classification)": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    "Heart Disease (Classification)": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/heart/heart.csv",
    "Mushroom (Classification)": "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
    "Digits (Classification)": "https://github.com/ageron/handson-ml2/raw/master/datasets/mnist/mnist.pkl.gz",
    "Spam Detection (Classification)": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/spambase.csv",
    "Cervical Cancer (Classification)": "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv",
    "Churn Prediction (Classification)": "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv",
    "NBA Players (Classification)": "https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/master/notebooks/data/nba.csv",
    "Mobile Price Classification": "https://www.kaggle.com/iabhishekofficial/mobile-price-classification/download"
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
