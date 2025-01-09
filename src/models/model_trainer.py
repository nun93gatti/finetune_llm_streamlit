import torch

from unsloth import FastLanguageModel, is_bfloat16_supported


class ModelTrainer:
    def __init__(self, model_name, config):
        self.model_name = "unsloth/" + model_name
        self.config = config
        self.load_model()

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.config["max_seq_length"],
            load_in_4bit=True,
            dtype=None,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "up_proj",
                "down_proj",
                "o_proj",
                "gate_proj",
            ],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )

    # def __init__(self, config):
    #     self.config = config
    #     self.model = None
    #     self.tokenizer = None

    # def load_model(self):
    #     """
    #     Load the base model and tokenizer
    #     """
    #     self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"])
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

    # def train(self, training_data):
    #     """
    #     Fine-tune the model
    #     """
    #     if self.model is None:
    #         self.load_model()

    #     training_args = TrainingArguments(
    #         output_dir=self.config["output_dir"],
    #         num_train_epochs=self.config["num_epochs"],
    #         per_device_train_batch_size=self.config["batch_size"],
    #         save_steps=self.config["save_steps"],
    #         logging_steps=self.config["logging_steps"],
    #     )

    #     trainer = Trainer(
    #         model=self.model, args=training_args, train_dataset=training_data
    #     )

    #     trainer.train()
