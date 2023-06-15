
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

class LLMFineTuner:
    def __init__(self, model_id, dataset_name):
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            load_in_4bit=True, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
        
        config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(model, config)
        self.model.print_trainable_parameters()

    def train(self, output_dir):
        dataset = load_dataset(self.dataset_name, split="train[:1000]")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            max_steps=100,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
        )
        
        trainer = Trainer(
            model=self.model,
            train_dataset=tokenized_dataset,
            args=training_args
        )
        trainer.train()

if __name__ == "__main__":
    tuner = LLMFineTuner("facebook/opt-350m", "wikitext")
    tuner.prepare_model()
    tuner.train("./lora-opt-results")
