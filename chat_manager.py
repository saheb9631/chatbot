# chat_manager.py

from sentiment_analyzer import SentimentAnalyzer
from google import genai
from google.genai import types
import os

class ChatManager:
    """
    Manages the conversation history, uses Gemini Chat Service for responses, 
    and integrates sentiment analysis. This class handles the core chat loop logic 
    and prepares data for the final report.
    """
    
    # System prompt to guide the Gemini Chat model's behavior
    SYSTEM_PROMPT = (
        "You are a conversational, empathetic, and highly efficient AI chatbot. "
        "Your responses should be brief, friendly, and directly address the user's "
        "latest statement, taking the conversation context into account. "
        "A user's sentiment is provided for context. Do not mention the sentiment "
        "label (Negative, Positive, Neutral) in your response, just react naturally "
        "to the emotional tone."
    )
    
    def __init__(self):
        self.history = [] 
        self.analyzer = SentimentAnalyzer()
        self.conversation_active = True
        
        # 1. Initialize Gemini Client (Key loaded via os.environ by dotenv)
        if 'GEMINI_API_KEY' not in os.environ:
             raise ValueError("GEMINI_API_KEY environment variable not set. Please set your key.")
             
        self.client = genai.Client()
        
        # 2. Initialize the dedicated Chat Service for better context management
        self.chat = self.client.chats.create(
            model='gemini-2.5-flash', # Use Flash for quick, cost-effective chatting
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT
            )
        )

    def get_bot_response(self, user_input: str, sentiment_label: str) -> str:
        """
        Generates a bot response using the Gemini Chat service.
        The user's sentiment is passed in for immediate context.
        """
        user_input_lower = user_input.lower()

        # 1. Exit Logic 
        if user_input_lower in ['bye', 'exit', 'quit', 'end']:
            self.conversation_active = False
            return "Thank you for sharing your thoughts! I'm ready to generate the sentiment report now."

        # 2. Craft the message content for the LLM
        # Integrating the sentiment result ensures the LLM reacts appropriately.
        message_to_llm = f"Sentiment detected: {sentiment_label}. User message: {user_input}"
        
        try:
            # 3. Send the message to the Gemini Chat service
            response = self.chat.send_message(message_to_llm)
            
            return response.text.strip()

        except Exception as e:
            # Robust error handling for API failures
            print(f"--- Gemini API Error ---")
            print(f"Encountered error: {e}")
            print(f"------------------------")
            return "I apologize, I've run into an issue processing your request. Could you try rephrasing that?"

    def process_user_message(self, user_message: str):
        """
        Performs statement-level sentiment analysis and stores the result locally.
        """
        sentiment_result = self.analyzer.analyze_statement(user_message)
        compound_score = sentiment_result['compound']
        sentiment_label = sentiment_result['label']
        
        # Store full history entry for local reporting (Tier 1/2 requirements)
        self.history.append({
            "speaker": "User",
            "message": user_message,
            "sentiment_score": compound_score,
            "sentiment_label": sentiment_label
        })
        
        # Tier 2 Requirement: Display statement-level output immediately
        print(f"\n[Statement Sentiment: **{sentiment_label}** (Score: {compound_score:.3f})]")

    def add_bot_message(self, bot_message: str):
        """
        Adds the bot's final displayed message to the local conversation history.
        """
        self.history.append({
            "speaker": "Bot",
            "message": bot_message,
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral"
        })

    # --- Reporting Methods (Tier 1 & Tier 2) ---

    def get_user_compound_scores(self) -> list:
        """Extracts only the compound scores from user messages."""
        return [item["sentiment_score"] for item in self.history if item["speaker"] == "User"]

    def generate_conversation_report(self) -> str:
        """Generates the final conversation report with Tier 1 & Tier 2 analysis."""
        user_scores = self.get_user_compound_scores()
        
        # --- TIER 1: Overall Conversation Sentiment (Retained for the calculation) ---
        overall_sentiment = self.analyzer.aggregate_sentiment(user_scores)
        report = f"\n\n{'='*70}\n## Conversation Analysis Report (LLM-Enhanced)\n{'='*70}\n"
        report += f"### TIER 1: Overall Emotional Direction (Calculated)\n"
        report += f"**Overall Sentiment:** **{overall_sentiment}**\n"
        
        if user_scores:
            avg_score = sum(user_scores) / len(user_scores)
            report += f"**Average Compound Score:** {avg_score:.3f}\n\n"

        # --- TIER 2 & 3: Structured LLM Analysis ---
        report += f"### TIER 2 & 3: Structured Conversation Analysis (Powered by Gemini-Pro)\n"
        
        if self.history:
            llm_report = self._generate_llm_summary()
            report += llm_report
        else:
            report += "No conversation history available to generate a detailed summary."
        
        report += f"\n{'='*70}\n"
        return report
    def _generate_llm_summary(self) -> str:
        """
        Uses Gemini-Pro to generate a structured, abstractive summary and analysis 
        based on the full conversation history and sentiment data.
        """
        # Format the history for the LLM, including sentiment data
        formatted_history = []
        for item in self.history:
            if item["speaker"] == "User":
                # Augment User messages with the classified sentiment
                formatted_history.append(
                    f"User (Sentiment: {item['sentiment_label']} / Score: {item['sentiment_score']:.3f}): {item['message']}"
                )
            else:
                formatted_history.append(f"Bot: {item['message']}")

        # The core prompt engineering template
        prompt_content = (
            "You are an expert Conversation Analyst AI. Your task is to analyze the following multi-turn conversation. "
            "The conversation history includes the sentiment label and score for each user turn. "
            "Generate a comprehensive report with three distinct sections:\n\n"
            "1. **Narrative Summary:** A concise, objective summary of the main topic and flow of the conversation.\n"
            "2. **Sentiment Analysis & Trend:** Describe the user's emotional journey. Note the peak positive/negative moments and explain any shifts (e.g., did a negative mood improve after a certain bot response?).\n"
            "3. **Conclusion & Recommendations (Tier 3 Insight):** Based on the flow, provide one specific, actionable recommendation for how the chatbot or a human agent could have handled a point in the conversation better, or what the user's ultimate goal or pain point was.\n\n"
            "CONVERSATION TRANSCRIPT:\n"
            f"{'\\n'.join(formatted_history)}"
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', # Use the same model for consistency
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    temperature=0.4 # Lower temperature for analytical, less creative output
                )
            )
            return response.text.strip()

        except Exception as e:
            return f"\n--- LLM Summary Error: Could not generate detailed report due to API issue: {e} ---"
    
