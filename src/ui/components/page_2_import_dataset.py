import streamlit as st
from models.model_trainer import ModelTrainer
from data.data_loader import DataLoader


def load_dataset(llm_obj: ModelTrainer):
    data_loader_obj = None
    st.title("Import Dataset")
    st.markdown("---")

    # Split main area into two columns
    col1, col2 = st.columns([1, 1])

    # Left column - Dataset Upload
    with col1:
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "parquet"])

        if uploaded_file is not None:
            with st.spinner(f"Loading and preprocessing data:..."):
                data_loader_obj = DataLoader(uploaded_file, llm_obj.tokenizer)
                data_loader_obj.load_training_data()
                print("done")

    # Right column - Display Preprocessed Dataset Format
    with col2:
        st.markdown("### Dataset Format After Preprocessing")
        if "preprocessed_df" in st.session_state:
            st.dataframe(data_loader_obj.dataset)

        else:
            st.info("Upload a dataset to see its format after preprocessing.")

    return data_loader_obj
