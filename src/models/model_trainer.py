from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer


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

    def train(self, data_obj):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=data_obj.tokenizer,
            train_dataset=data_obj.dataset,
            dataset_text_field="text",
            max_seq_length=self.config["max_seq_length"],
            dataset_num_proc=1,
            packing=True,
            args=TrainingArguments(
                learning_rate=self.config["learning_rate"],
                lr_scheduler_type="linear",
                per_device_train_batch_size=self.config["per_device_train_batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                num_train_epochs=2,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim=self.config["optim"],
                weight_decay=self.config["weight_decay"],
                warmup_steps=self.config["warmup_steps"],
                output_dir="output",
                seed=0,
            ),
        )

        trainer.train()
