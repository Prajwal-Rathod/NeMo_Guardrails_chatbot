# NeMo Guardrails Chatbot (Groq API Version)

A conversational AI chatbot powered by [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) and enhanced with the **Groq API** for ultra-fast, low-latency responses.

This project demonstrates how to integrate **Colang-based dialogue flows**, **custom actions**, and a **Groq-backed LLM** into a safety-aware chatbot.

---

## 🚀 Features

- ✅ Configurable with **Colang** rules for flow control
- ⚡ Backed by **Groq API** for fast LLM inference
- 🛡️ Includes **Guardrails** to restrict and shape conversations
- 🧩 Easily customizable actions and flows
- 🖥️ CLI-based chatbot interface

---

## 📂 Project Structure

```
NeMo_Guardrails_chatbot/
├── config/
│   ├── actions.py        # Python functions triggered by Colang
│   ├── config.yml        # NeMo config - uses Groq API and defines rails
│   └── flows.co          # Main Colang dialogue flow
├── launch_app.py         # Entry point to run the chatbot
├── requirements.txt      # Python dependencies
└── .gitignore            # Git ignore rules
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Prajwal-Rathod/NeMo_Guardrails_chatbot.git
cd NeMo_Guardrails_chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your environment

Create a `.env` file or export your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Ensure this key is available in your shell or `.env`.

---

## 🧠 How It Works

- The chatbot logic is written using **Colang** (`flows.co`)
- When the user sends a message, it's matched against rules in Colang
- If needed, custom Python `actions.py` are invoked
- All LLM responses are generated via **Groq API** (e.g., `llama3-8b-8192`)

---

## ▶️ Run the Chatbot

Simply execute:

```bash
python launch_app.py
```

You'll enter a terminal-based chat interface powered by NeMo Guardrails + Groq API.

## 🤖 Using Groq API

In `config.yml`:

```yaml
models:
  - type: main
    engine: groq
    model: llama3-8b-8192
    api_key: ${GROQ_API_KEY}
```

This ensures all chatbot replies are powered by **Groq LLMs** (like LLaMA 3).

---

## 📌 Notes

- ✅ NeMo version: `0.14.x`
- ✅ Groq model supported: `llama3-8b-8192`.
- ❗ Make sure your Groq account has sufficient quota

---

## 💡 Suggestions

- Add more conversation flows using `flows.co`
- Integrate external APIs via `actions.py` (e.g., weather, search)
- Deploy with Gradio or Streamlit for UI
