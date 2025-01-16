import pandas as pd
from datasets import Dataset
from unsloth.chat_templates import get_chat_template


# def load_dataset(uploaded_file):
#     df = pd.read_csv(uploaded_file)


# def preprocess_dataset(df):
#     # Placeholder for preprocessing logic
#     # Replace this with actual preprocessing steps
#     # For example, you might clean data, handle missing values, etc.
#     return df.head()  # Display the first few rows as a sample


class DataLoader:
    def __init__(self, raw_input_dataset, tokenizer):
        self.raw_input_dataset = raw_input_dataset
        self.tokenizer = tokenizer
        self.load_training_data()

    # Apply the template function
    def apply_template(self, examples):
        messages = examples["conversations"]
        text = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )
            for message in messages
        ]
        return {"text": text}

    def preprocess_dataset(self):
        self.tokenizer = get_chat_template(
            self.tokenizer,
            mapping={
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            chat_template="chatml",
        )

    def load_training_data(self):
        """
        Load and prepare training data
        """

        # Load the Parquet file into a DataFrame
        df = pd.read_parquet(self.raw_input_dataset)[:10]

        # Convert the DataFrame to a Hugging Face Dataset
        self.dataset = Dataset.from_pandas(df)
        self.preprocess_dataset()

        self.dataset = self.dataset.map(self.apply_template, batched=True)
