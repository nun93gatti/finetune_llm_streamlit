import streamlit as st
from models.model_trainer import ModelTrainer
from utils.input_data import preprocess_dataset


def load_dataset(llm_obj: ModelTrainer = None):
    st.title("Import Dataset")
    st.markdown("---")

    # Split main area into two columns
    col1, col2 = st.columns([1, 1])

    # Left column - Dataset Upload
    with col1:
        st.markdown("### Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            with st.spinner(f"Loading and preprocessing data:..."):
                preprocessed_df = load_dataset(uploaded_file)
            st.session_state["preprocessed_df"] = preprocessed_df

    # Right column - Display Preprocessed Dataset Format
    with col2:
        st.markdown("### Dataset Format After Preprocessing")
        if "preprocessed_df" in st.session_state:
            st.dataframe(st.session_state["preprocessed_df"])
        else:
            st.info("Upload a dataset to see its format after preprocessing.")
