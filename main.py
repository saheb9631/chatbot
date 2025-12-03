# main.py

import os
from dotenv import load_dotenv
load_dotenv()

import nltk
from chat_manager import ChatManager

def setup():
    """Ensure all required libraries and data are available."""
    
    # Check for the critical API key before proceeding
    if 'GEMINI_API_KEY' not in os.environ:
         print("CRITICAL ERROR: The GEMINI_API_KEY environment variable is not set.")
         print("Please set it before running the chatbot.")
         exit(1)
         
    # Ensure NLTK data is downloaded (if using any NLTK functionality, though 
    # VADER is replaced, keeping this check is good practice if any other 
    # NLTK dependencies are introduced later).
    try:
        nltk.data.find('corpora/stopwords') # A common small NLTK resource
        print("NLTK resources found.")
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    print("Setup complete. Initializing models...")


def run_chatbot():
    """Main execution loop for the chatbot interaction."""
    
    try:
        manager = ChatManager()
    except Exception as e:
        print(f"FATAL ERROR during ChatManager initialization: {e}")
        return # Stop execution if initialization fails

    print("\n" + "="*70)
    print("Welcome to the Industry-Grade Sentiment Analysis Chatbot! (Gemini-Pro & RoBERTa)")
    print("Type 'exit' or 'quit' to end the conversation and generate the report.")
    print("="*70 + "\n")

    while manager.conversation_active:
        try:
            user_input = input("You: ")
            
            if not user_input.strip():
                continue

            # 1. Process User Message & Tier 2 Analysis
            manager.process_user_message(user_input)
            
            # Get the sentiment label from the latest history entry
            current_sentiment = manager.history[-1]['sentiment_label']
            
            # 2. Get and Display Bot Response (using the sentiment for context)
            bot_response = manager.get_bot_response(user_input, current_sentiment)
            manager.add_bot_message(bot_response)
            
            print(f"Bot: {bot_response}")

        except EOFError:
            manager.conversation_active = False
            print("\nChat ended abruptly. Generating report.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    # After the loop ends, generate the final report
    final_report = manager.generate_conversation_report()
    print(final_report)


if __name__ == "__main__":
    setup()
    run_chatbot()