# README

## Project Overview

This project fine-tunes the **Meta-LLaMA-3-8B** model for a regression task. The regression model predicts numerical values from input text by attaching a regression head to the pretrained model. The script is designed to preprocess data, train the model, and perform inference efficiently.

## Code Structure

The code is divided into the following sections:

### 1. **Environment Setup**
- Install required libraries.
- Log in to Hugging Face Hub to access pretrained models.

### 2. **Model Preparation**
- Load the **Meta-LLaMA-3-8B** model and tokenizer.
- Add padding tokens if not already present.
- Freeze the base model's parameters to focus only on the regression head.

### 3. **Custom Model Definition**
- A custom `RegressionModel` class extends `nn.Module` to include a regression head.

### 4. **Dataset Handling**
- Load the Yelp Polarity dataset using Hugging Face's `datasets` library.
- Preprocess text data to tokenize inputs and assign dummy regression labels.

### 5. **Training**
- Use Hugging Face's `Trainer` for model training with custom loss computation.

### 6. **Evaluation and Saving**
- Evaluate the fine-tuned model.
- Save the trained model, tokenizer, and model weights locally.

### 7. **Inference**
- Define a `RegressionPredictor` class for loading the model and performing predictions.

---


