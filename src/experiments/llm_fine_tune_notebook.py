import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
import unsloth
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import xformers

import pandas as pd
from datasets import Dataset

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
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


# Load the Parquet file into a DataFrame
df = pd.read_parquet(r"C:\Users\CAMNG3\Downloads\train-00000-of-00001.parquet")[:10]

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)


# Apply the template function
def apply_template(examples):
    messages = examples["conversations"]
    text = [
        tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        for message in messages
    ]
    return {"text": text}


# Map the function to the dataset
dataset = dataset.map(apply_template, batched=True)

print("Ciao")

# Now you can use the dataset as needed

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()


model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
outputs = model.generate(
    input_ids=inputs, streamer=text_streamer, max_new_tokens=100, use_cache=True
)
predictions = tokenizer.batch_decode(
    outputs[:, inputs.shape[1] :], skip_special_tokens=True
)

print(f"Answer is : {predictions}")
