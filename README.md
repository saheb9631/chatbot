# Conversational AI With Tiered Emotional Intelligence  
*A Hybrid LLM + Sentiment Intelligence System*  
**Author: Saheb Ansari**

---

## 📘 Introduction
This project delivers a next-generation conversational AI system that goes far beyond simple text replies. By combining a **fine-tuned RoBERTa sentiment classifier** with **Google’s Gemini 2.5 Flash LLM**, the system performs:

- Real-time empathetic conversation  
- Deep emotional diagnostics  
- Multi-layer behavioral reasoning  
- End-of-session intelligence reporting for business insights  

This hybrid design enables both **human-like responses** and **high-value analytical output**, ideal for customer support, mental health platforms, enterprise automation, and conversational analytics.

---

## ⭐ Why This Model Is Superior
### 1. Deeper Contextual Empathy
RoBERTa provides **pre-validated sentiment labels** and confidence scores, letting Gemini focus on generating strategic, empathetic responses.

### 2. Tiered Business Intelligence
- **Tier 1 – Sentiment Classification**  
- **Tier 2 – Emotion-aware Real-time Responses**  
- **Tier 3 – Full-Conversation Diagnostic Report**  

The system outputs summaries, trend analysis, and actionable recommendations.

### 3. Cost-Optimized, Low-Latency Architecture
Gemini Flash ensures:
- High throughput  
- Fast real-time responses  
- Strong reasoning for summaries  

---

## 🧠 Technologies Used
- **RoBERTa-base (social media fine-tuned)**  
- **Google Gemini 2.5 Flash (Google GenAI SDK)**  
- **Structured System Prompting**  

---

## 🔄 End-to-End Model Flow
### Step 1 — Tier 1 Analysis
User message → RoBERTa outputs sentiment → stored in history.

### Step 2 — Tier 2 Real-Time Response
Sentiment and message are included in Gemini prompt → empathetic reply.

### Step 3 — Tier 3 Post-Conversation Analysis
Full annotated conversation is sent to Gemini → narrative summary + trend map + actionable insights.

---

## 📦 Inferred Dependencies
```
transformers
torch
google-genai
python-dotenv
fastapi
uvicorn
pydantic
numpy
```

---

## 🛠️ Installation
```bash
pip install transformers torch google-genai python-dotenv fastapi uvicorn
```

---

## ▶️ Usage Example
```python
sentiment = roberta_model.predict(user_input)

prompt = "Sentiment detected: {} (Score: {})\nUser message: {}".format(
    sentiment.label, sentiment.score, user_input
)

response = gemini.generate(prompt)
```

---

## 🧩 System Architecture
User → RoBERTa → Gemini Flash → Response  
      ↓  
    Tier 3 Analytics  

---

## 🧪 Features
- Hybrid LLM + ML pipeline  
- Emotion-aware conversational agent  
- Session-level business intelligence  
- Low latency, production-ready design  

---

## 🐛 Troubleshooting
| Issue | Cause | Fix |
|-------|--------|------|
| Slow responses | No streaming | Enable Gemini streaming |
| Wrong sentiment | Insufficient tuning | Re-train RoBERTa |
| Poor summaries | Weak prompts | Strengthen Tier 3 instructions |

---

## 👤 Author
**Saheb Ansari**

---


