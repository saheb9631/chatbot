# sentiment_analyzer.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch
import os

class SentimentAnalyzer:
    """
    Performs sentiment analysis using the cardiffnlp/twitter-roberta-base-sentiment-latest model
    with device optimization (GPU preferred, falls back to CPU).
    """
    def __init__(self):
        # Determine the device for model computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sentiment model running on: {self.device}")

        # 1. Load Tokenizer and Model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Load the model directly onto the determined device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set model to evaluation mode (essential for inference)
        
        # RoBERTa output labels
        self.labels = ['Negative', 'Neutral', 'Positive'] 
        
    def analyze_statement(self, text: str) -> dict:
        """
        Performs sentiment analysis on a single piece of text.
        """
        # Tokenize the input text and send tensors to the correct device
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs (logits) without calculating gradients (memory/speed optimization)
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Apply softmax to convert logits to probabilities
        scores = F.softmax(output.logits, dim=1).squeeze().cpu() # Move scores back to CPU for standard ops
        
        # Determine the predicted label
        predicted_index = torch.argmax(scores).item()
        predicted_label = self.labels[predicted_index]
        
        # Calculate compound score (Pos score - Neg score)
        compound_score = scores[2].item() - scores[0].item()
        
        return {
            'compound': compound_score,
            'label': predicted_label
        }

    def aggregate_sentiment(self, compound_scores: list) -> str:
        """
        Calculates the average compound score for a list of scores and returns 
        the overall sentiment label for the conversation.
        """
        if not compound_scores:
            return "Neutral (No user input)"
        
        avg_score = sum(compound_scores) / len(compound_scores)
        
        # Use slightly wider thresholds for robust aggregated sentiment
        if avg_score >= 0.35:
            return "Positive"
        elif avg_score <= -0.35:
            return "Negative"
        else:
            return "Neutral"