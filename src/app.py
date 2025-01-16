import streamlit as st

from utils.config import load_model_configs
from ui.components.page_1_choose_llm_page import choose_llm_page
from ui.components.page_2_import_dataset import load_dataset
from ui.components.page_3_training_LLM import fine_tuning_LLM


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
        st.session_state["llm_obj"] = llm_obj
        st.session_state["llm_selected"] = llm_obj is not None
    elif selected_page == "Import Dataset":
        if st.session_state.get("llm_selected", False):
            data_loader_obj = load_dataset(st.session_state["llm_obj"])
            st.session_state["input_selected"] = data_loader_obj is not None
            st.session_state["preprocessed_df"] = data_loader_obj
        else:
            st.warning("You need to select an LLM before proceeding.")
    elif selected_page == "Training":
        if st.session_state.get("preprocessed_df", False) and st.session_state.get(
            "llm_selected", False
        ):
            fine_tuning_LLM(
                st.session_state["llm_obj"],
                st.session_state["preprocessed_df"],
            )

        else:
            st.warning(
                "You need to select an LLM and import a dataset before proceeding."
            )
        # Add your training configuration logic here

    elif selected_page == "Inference":
        st.title("Inference")
        st.info("Inference functionality will be implemented here.")
        # Add your inference logic here


if __name__ == "__main__":
    main()
