# -*- coding: utf-8 -*-

!pip install transformers accelerate datasets

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import login, whoami

# Below is the inference phase. We can download the pre-trained model(we trained above) to use. 
# I am thinking about maybe storing this weights file in S3 or another cloud storage platform.
# In this code when inference it still downloads the pre-trained model again to get its model structure,
# However in the future maybe we can download only structure (exclude its weights), save more RAM during inference, and increase the batch size

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
