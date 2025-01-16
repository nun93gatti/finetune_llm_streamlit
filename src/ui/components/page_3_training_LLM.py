from data.data_loader import DataLoader
from models.model_trainer import ModelTrainer
import streamlit as st
from io import StringIO
from contextlib import redirect_stdout
import time


def fine_tuning_LLM(llm_obj: ModelTrainer, data_loader_obj: DataLoader):
    st.title("Training Setup")
    st.markdown("---")

    # Check if LLM object is loaded
    if "llm_obj" in st.session_state:
        st.markdown("### Loaded LLM Object")
        st.write(llm_obj.model_name)
    else:
        st.warning("No LLM object loaded. Please select an LLM first.")

    # Check if preprocessed DataFrame is loaded
    if "preprocessed_df" in st.session_state:
        st.markdown("### Preprocessed Dataset")
        st.dataframe(data_loader_obj.dataset)
    else:
        st.warning("No preprocessed dataset loaded. Please import a dataset first.")

    log_placeholder = st.empty()

    # Start Training Button
    if st.button("Start Training"):
        with StringIO() as buf, redirect_stdout(buf):
            llm_obj.train(data_loader_obj)
            while True:
                # Update the log placeholder with the current buffer content
                log_placeholder.text(buf.getvalue())
                time.sleep(0.1)  # Adjust the sleep time as needed
    else:
        st.info("Click the button to start the training")
