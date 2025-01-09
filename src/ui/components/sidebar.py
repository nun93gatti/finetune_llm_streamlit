import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.title("�� LLM Fine-tuning")

        selected_page = st.radio(
            "Navigation", options=["Home", "Data Preparation", "Training"], index=0
        )

        st.divider()

        st.markdown("### Model Settings")
        model_name = st.selectbox(
            "Base Model",
            options=[
                "EleutherAI/pythia-70m",
                "EleutherAI/pythia-160m",
                "EleutherAI/pythia-410m",
            ],
            index=0,
        )

        st.markdown("### Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 10, 3)
        batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16, 32], value=4)

    return selected_page
