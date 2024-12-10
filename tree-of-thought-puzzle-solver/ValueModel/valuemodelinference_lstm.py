

import torch
import torch.nn as nn
from transformers import AutoTokenizer


class LSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(LSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return self.activation(output)


class LSTMPredictor:
    def __init__(self):
        vocab_size = 30522
        embedding_dim = 128
        hidden_dim = 64
        output_dim = 1
        pad_idx = 0
        model_save_path = "lstm_regressor.pth"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_model = LSTMRegressor(
            vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx).to(self.device)
        self.loaded_model.load_state_dict(torch.load(model_save_path))
        self.loaded_model.eval()
        print("Model loaded successfully.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")

    def predict(self, input_texts, max_len=50):
        """
        Predict the q_value for given Sudoku states using the trained model.

        Args:
            input_texts: List of Sudoku states as strings.
            max_len: Maximum sequence length for tokenization.

        Returns:
            List of predicted q_values.
        """

        inputs = self.tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            predictions = self.loaded_model(input_ids, attention_mask)
            return predictions.squeeze().tolist()


# example_states = [

#     "[['1', '3', '2'], ['2', '1', '3'], ['3', '2', '1']]",

#     "[['1', '1', '1'], ['1', '1', '1'], ['1', '1', '1']]",

#     "[['*', '3', '2'], ['2', '*', '*'], ['3', '*', '1']]",

#     "[['1', '3', '2'], ['2', '1', '3'], ['3', '*', '*']]",

#     "[['3', '2', '1'], ['3', '2', '1'], ['3', '2', '1']]",

#     "[['1', '*', '*'], ['*', '2', '*'], ['*', '*', '3']]",

#     "[['3', '1', '2'], ['2', '3', '1'], ['1', '2', '3']]",

#     "[['0', '0', '0'], ['0', '0', '0'], ['0', '0', '0']]",

#     "[['1', '*', '2'], ['*', '1', '*'], ['2', '*', '3']]",

#     "[['*', '*', '1'], ['*', '2', '*'], ['3', '*', '*']]"
# ]


# predicted_q_values = predict_q_value(loaded_model, tokenizer, example_states)
# for state, q_value in zip(example_states, predicted_q_values):
#     print(f"State: {state}")
#     print(f"Predicted Q-Value: {q_value:.4f}")
