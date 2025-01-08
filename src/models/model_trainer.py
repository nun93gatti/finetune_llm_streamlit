from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the base model and tokenizer
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

    def train(self, training_data):
        """
        Fine-tune the model
        """
        if self.model is None:
            self.load_model()

        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            save_steps=self.config["save_steps"],
            logging_steps=self.config["logging_steps"],
        )

        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=training_data
        )

        trainer.train()
