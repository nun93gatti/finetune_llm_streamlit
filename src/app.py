import streamlit as st

from utils.config import load_model_configs, save_model_configs
from ui.components.choose_llm_page import choose_llm_page
from ui.components.import_dataset import load_dataset


def main():
    st.set_page_config(page_title="LLM Fine-tuning App", layout="wide")

    # Load custom CSS
    with open("src/ui/styles/main.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load model configurations
    if "model_configs" not in st.session_state:
        st.session_state.model_configs = load_model_configs()

    # Navigation Sidebar with enhanced styling
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">🤖 Navigation</div>', unsafe_allow_html=True
        )

        # Navigation menu with custom styling
        selected_page = st.radio(
            "",
            ["Choose LLM", "Import Dataset", "Training", "Inference"],
            format_func=lambda x: f"{'🏠' if x == 'Welcome' else '📂' if x == 'Import Dataset' else '⚙️' if x == 'Training' else '🔍'} {x}",
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

    # Main content area
    if selected_page == "Choose LLM":
        llm_obj = choose_llm_page(st.session_state.model_configs)
    elif selected_page == "Import Dataset":
        load_dataset("")
        # Add your dataset import logic here

    elif selected_page == "Training":
        st.title("Training Configuration")
        st.info("Training configuration will be implemented here.")
        # Add your training configuration logic here

    elif selected_page == "Inference":
        st.title("Inference")
        st.info("Inference functionality will be implemented here.")
        # Add your inference logic here


if __name__ == "__main__":
    main()
