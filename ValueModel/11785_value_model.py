# -*- coding: utf-8 -*-

!pip install transformers accelerate datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login, whoami

login("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(whoami())

# dataset = load_dataset("yelp_polarity")

# print(dataset["train"][0:5])
# dataset = dataset["train"][0:100]



model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    model.resize_token_embeddings(len(tokenizer))


for param in model.parameters():
    param.requires_grad = False

class RegressionModel(nn.Module):
    def __init__(self, base_model):
        super(RegressionModel, self).__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        regression_output = self.regression_head(pooled_output)
        return regression_output.squeeze(-1)

model = RegressionModel(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


dataset = load_dataset("yelp_polarity")


print(dataset["train"][0:5])


small_train_dataset = dataset["train"].select(range(100))
small_test_dataset = dataset["test"].select(range(50))


def preprocess_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = [float(len(text) % 5) for text in examples["text"]]
    return inputs


tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        labels = labels.to(self.args.device).float()
        outputs = model(**inputs)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)


trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset
)


trainer.train()


results = trainer.evaluate()


model.save_pretrained("llama3_regression_model")
tokenizer.save_pretrained("llama3_regression_tokenizer")

torch.save(model.state_dict(), "llama3_regression_model.pth")
tokenizer.save_pretrained("llama3_regression_tokenizer")

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class RegressionPredictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("llama3_regression_tokenizer")

        model_name = "meta-llama/Meta-Llama-3-8B"
        self.base_model = AutoModel.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        class RegressionModel(nn.Module):
            def __init__(self, base_model):
                super(RegressionModel, self).__init__()
                self.base_model = base_model
                self.regression_head = nn.Linear(base_model.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                regression_output = self.regression_head(pooled_output)
                return regression_output.squeeze(-1)


        self.model = RegressionModel(self.base_model)

        self.model.load_state_dict(torch.load("llama3_regression_model.pth"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_texts):
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_values = outputs.cpu().numpy().tolist()

        return predicted_values

#predictor = RegressionPredictor()
input_texts = ["This product exceeded my expectations.", "Not worth the price."]
predicted_values = predictor.predict(input_texts)
print("Predicted Values:", predicted_values)
