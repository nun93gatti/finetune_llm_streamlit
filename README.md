# LLM Fine-Tuning Streamlit App

This Streamlit application allows users to fine-tune a language model (LLM) using custom datasets. The app provides an interactive interface for selecting a pre-trained model, importing datasets, and initiating the training process.

## Features

- **Model Selection**: Choose from a list of pre-trained language models.
- **Dataset Import**: Upload datasets in CSV or Parquet format for preprocessing.
- **Training Setup**: Configure and start the fine-tuning process with real-time log updates.
- **Inference**: (Planned) Perform inference using the fine-tuned model.

## Prerequisites

- Python 3.11 or higher
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/) (or another deep learning framework, depending on your model)
- Other dependencies as listed in `requirements.txt`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nun93gatti/finetune_llm_streamlit
   cd your-repo-name
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. **Start the Streamlit App**

   ```bash
   streamlit run src/app.py
   ```

2. **Access the App**

   Open your web browser and go to `http://localhost:8501` to interact with the app.

## Usage

1. **Choose LLM**: Navigate to the "Choose LLM" page to select a pre-trained language model.
2. **Import Dataset**: Go to the "Import Dataset" page to upload your dataset in CSV or Parquet format.
3. **Training Setup**: Use the "Training Setup" page to review the loaded model and dataset, then click "Start Training" to begin fine-tuning.
4. **Inference**: (Planned) Use the "Inference" page to test the fine-tuned model.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

