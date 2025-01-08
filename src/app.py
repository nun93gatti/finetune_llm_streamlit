import streamlit as st
import yaml
from pathlib import Path


def load_model_configs():
    config_path = Path("configs/model_configs.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_model_configs(configs):
    config_path = Path("configs/model_configs.yaml")
    with open(config_path, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)


def main():
    st.set_page_config(page_title="LLM Fine-tuning App", layout="wide")

    # Load custom CSS
    with open("ui/styles/main.css") as f:
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
            ["Welcome", "Training"],
            format_func=lambda x: f"{'🏠' if x == 'Welcome' else '⚙️'} {x}",
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

    # Main content area
    if selected_page == "Welcome":
        st.title("LLM Fine-tuning")
        st.markdown("---")

        # Split main area into two columns
        col1, col2 = st.columns([1, 1])

        # Left column - Model Selection
        with col1:
            st.markdown("### Model Selection")
            selected_model = st.selectbox(
                "Choose a model",
                options=["model_" + x for x in "abcde"],
                format_func=lambda x: x.replace("model_", "Model ").upper(),
            )

        # Right column - Model Configuration
        with col2:
            if selected_model in st.session_state.model_configs:
                config = st.session_state.model_configs[selected_model]

                st.markdown("### Model Configuration")

                # Editable configuration values
                new_weight_decay = st.number_input(
                    "Weight Decay",
                    value=float(config["weight_decay"]),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.001,
                    format="%.3f",
                )

                new_optim = st.selectbox(
                    "Optimizer",
                    options=["adamw_8bit", "adam", "sgd"],
                    index=["adamw_8bit", "adam", "sgd"].index(config["optim"]),
                )

                # Save button for configuration changes
                if st.button("Save Configuration"):
                    st.session_state.model_configs[selected_model].update(
                        {"weight_decay": new_weight_decay, "optim": new_optim}
                    )
                    save_model_configs(st.session_state.model_configs)
                    st.success("Configuration saved successfully!")

    elif selected_page == "Training":
        st.title("Training Configuration")
        st.info("Training configuration will be implemented here")


if __name__ == "__main__":
    main()
