import streamlit as st
from utils.config import save_model_configs, load_model
from models.model_trainer import ModelTrainer


def choose_llm_page(model_configs):
    llm_obj = None
    st.title("LLM Fine-tuning")
    st.markdown("---")

    # Extract model names from the configuration
    model_names = list(model_configs.keys())

    # Split main area into two columns
    col1, col2 = st.columns([1, 1])

    # Left column - Model Selection
    with col1:
        st.markdown("### Model Selection")
        selected_model = st.selectbox(
            "Choose a model",
            options=model_names,  # Use model names from config
            format_func=lambda x: x.replace("-", " ").title(),
        )

        # Display the model description from the configuration
        if selected_model in model_configs:
            description = model_configs[selected_model].get(
                "description", "No description available."
            )
            st.markdown(f"**Model Description:** {description}")

        # Add vertical space before the Load Model button
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Load Model button under Model Selection
        if st.button("Load Model"):
            llm_obj = load_model(selected_model)

    # Right column - Model Configuration
    with col2:
        if selected_model in model_configs:
            config = model_configs[selected_model]

            st.markdown("### Model Configuration")

            # Iterate over each configuration item
            updated_config = {}
            for key, value in config.items():
                # Skip the description key to ensure it doesn't appear in the configuration section
                if key == "description":
                    continue

                if isinstance(value, float):
                    updated_value = st.number_input(
                        key.replace("_", " ").title(),
                        value=value,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.0001,  # Adjust step for precision
                        format="%.4f",  # Adjust format for four decimal places
                    )
                elif isinstance(value, int):
                    updated_value = st.number_input(
                        key.replace("_", " ").title(),
                        value=value,
                        min_value=0,
                        step=1,
                        format="%d",
                    )
                elif isinstance(value, str):
                    updated_value = st.text_input(
                        key.replace("_", " ").title(),
                        value=value,
                    )
                else:
                    continue  # Skip unsupported types

                updated_config[key] = updated_value

            # Save button for configuration changes
            if st.button("Save Configuration"):
                model_configs[selected_model].update(updated_config)
                save_model_configs(model_configs)
                st.success("Configuration saved successfully!")

    return llm_obj
