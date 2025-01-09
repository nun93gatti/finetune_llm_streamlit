import yaml
from pathlib import Path
import streamlit as st
from models.model_trainer import ModelTrainer


def load_model_configs():
    config_path = Path("configs/model_configs.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_model_configs(configs):
    config_path = Path("configs/model_configs.yaml")
    with open(config_path, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)


def load_model(selected_model):
    with st.spinner(f"Loading model: {selected_model}..."):
        # Placeholder for model loading logic
        # Replace this with actual model loading code
        # Example: model = load_your_model_function(selected_model)
        # st.session_state['loaded_model'] = model
        llm_model = ModelTrainer(
            selected_model, st.session_state.model_configs[selected_model]
        )
    # time.sleep(2)  # Simulate a delay for loading
    st.success(f"Model {selected_model} loaded successfully!")

    return llm_model
