# app.py (verbose / debug friendly)
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import traceback
import logging
import sys

# simple console logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Try to import your ChatManager and show clear errors if any
manager = None
try:
    logging.info("Attempting to import ChatManager from chat_manager.py ...")
    from chat_manager import ChatManager
    logging.info("ChatManager imported successfully.")
    try:
        logging.info("Initializing ChatManager() (this may take time)...")
        manager = ChatManager()
        logging.info("ChatManager initialized.")
    except Exception as e:
        logging.error("ChatManager initialization failed: %s", e)
        logging.error(traceback.format_exc())
        manager = None
except Exception as e:
    logging.error("Failed to import ChatManager: %s", e)
    logging.error(traceback.format_exc())
    manager = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if manager is None:
        return jsonify({"error": "Server not ready. ChatManager not initialized. Check server logs."}), 500
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        manager.process_user_message(user_message)
        last = manager.history[-1]
        sentiment_label = last.get("sentiment_label")
        sentiment_score = last.get("sentiment_score")

        bot_reply = manager.get_bot_response(user_message, sentiment_label)
        manager.add_bot_message(bot_reply)

        return jsonify({
            "bot": bot_reply,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score
        })
    except Exception as e:
        logging.error("Error in /api/chat: %s", e)
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/report", methods=["GET"])
def api_report():
    if manager is None:
        return jsonify({"error": "Server not ready. ChatManager not initialized. Check server logs."}), 500
    try:
        report = manager.generate_conversation_report()
        return jsonify({"report": report})
    except Exception as e:
        logging.error("Error in /api/report: %s", e)
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server on http://0.0.0.0:8501 ...")
    try:
        app.run(host="0.0.0.0", port=8501, debug=False)
    except Exception as e:
        logging.error("Failed to start Flask server: %s", e)
        logging.error(traceback.format_exc())
