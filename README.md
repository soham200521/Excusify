# Excusify
**AI-powered excuse, apology & emergency simulator with proof generation and feedback intelligence**

---

## 🧠 Overview

This project is an all-in-one **Generative AI system** that creates realistic excuses, professional apologies, and emergency messages — complete with **auto-generated PDF proofs, WhatsApp/SMS chat visuals, voice output**, and **behavior-predictive feedback learning**.

Built for **real-life simulation**, the system goes beyond simple text generation to include audio playback, multilingual output, location logs, and a smart dashboard that evolves with user interaction.

---

## Table of Contents

- [Overview](#-overview)
- [Features at a Glance](#-features-at-a-glance)
- [Custom Datasets](#-custom-datasets)
- [AI Models & Training](#-ai-models--training)
- [Real-Life Simulation Outputs](#-real-life-simulation-outputs)
- [Smart Feedback & Dashboard](#-smart-feedback--dashboard)
- [Tech Stack](#-tech-stack)
- [Usage](#%EF%B8%8F-usage)
- [Project Structure](#%EF%B8%8F-project-structure)
- [License](#-license)
- [Created By](#-created-by)

---

## 🚀 Features at a Glance

| Feature | Description |
|--------|-------------|
| ✍️ Excuse Generator | Fine-tuned GPT-2 model trained on a custom dataset to produce believable, context-aware excuses |
| 🚨 Emergency Messages | AI-generated alerts for urgent scenarios (work, school, family, social) with SMS-style visuals & PDF reports |
| 🙏 Apology Writer | Supports *emotional* and *professional* apology styles with WhatsApp-like chat output |
| 📄 Proof Documents | Automatically generate official-looking PDFs, WhatsApp chats, and location logs as supporting "evidence" |
| 🗣️ Voice Output | Converts generated text to natural-sounding speech using `gTTS` |
| 🌐 Multilingual Support | Translates all outputs into 9+ languages (Hindi, French, Japanese, etc.) |
| 📊 Feedback & Ranking | Learns from user feedback to rank future excuses and predict excuse-need behavior |
| 🔒 Parental Lock | Filters out inappropriate or sensitive content automatically |

---

## 📂 Custom Datasets

All core models are trained on **hand-crafted datasets** tailored for realistic use-cases:

| Dataset | Purpose |
|---------|---------|
| `intelligent_excuses_dataset.csv` | Scenario-driven excuse samples (scenario, urgency, believability, excuse) |
| `guilt_tripping_apologies.csv` | Apology types: emotional and professional |
| `emergency_messages.csv` | Realistic emergency messages across 4 scenarios |
| `excuse_history.csv` | Feedback logs (likes, ratings, favorites, timestamps) auto-generated during use |

---

## 🤖 AI Models & Training

Each content type is powered by a **fine-tuned GPT-2 model** via Hugging Face Transformers:

- `gpt2-finetuned-excuses`
- `gpt2-finetuned-apologies`
- `gpt2-finetuned-emergency`

**Training Details:**
- Optimizer: AdamW  
- Epochs: 3  
- Token Length: 64  
- Batch Size: 4  
- Device: CUDA (if available)  
- Platform: Google Colab

---

## 💬 Real-Life Simulation Outputs

| Proof Type | Description |
|------------|-------------|
| 📱 WhatsApp-style Chat | AI-generated excuses/apologies shown in chat UI |
| ✉️ SMS Emergency Chat | Auto-formatted emergency messages with sender name |
| 📄 PDF Certificate | Official-sounding statements with date, signature & headers |
| 📍 Location Logs | Fake GPS trails with timestamps and addresses |
| 🔊 Audio Messages | Natural voice messages in user's selected language |

---

## 📊 Smart Feedback & Dashboard

The system **learns from user behavior** to improve itself:
- Ranks generated excuses using `RandomForestRegressor` based on past ratings
- Predicts if user might need an excuse using `DecisionTreeClassifier` (based on time/day patterns)
- Dashboard shows total usage, most-liked excuse, average rating & more

---

## 🛠 Tech Stack

- 🧠 Hugging Face Transformers (GPT-2)
- 🐍 Python (Pandas, NumPy, Scikit-learn)
- 🎨 Pillow + ReportLab (visual/image generation)
- 🌐 Googletrans (language translation)
- 🔊 gTTS (text-to-speech)
- 🎲 Faker (location logs)
- 💻 Google Colab + IPyWidgets (UI)

---

## ▶️ Usage

- Launch the app and select a mode (Excuse, Apology, Emergency).
- Fill in the prompt fields and generate your output.
- Download PDF, listen to audio, or view chat visualizations.
- Provide feedback to help the AI improve.

---

## 🗂️ Project Structure
```
Excusify/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/
  ├── intelligent_excuses_dataset.csv
  ├── guilt_tripping_apologies.csv
  └── emergency_messages.csv
```

---

## 🪪 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## 🤗 Models

This app uses fine-tuned GPT-2 models hosted on the Hugging Face Hub.  
You do **not** need to download the models manually - when you run the app, it will automatically load each model from the Hub.

- [Excuse Generator Model](https://huggingface.co/Sohamb2005/gpt2-finetuned-excuses)
- [Apology Generator Model](https://huggingface.co/Sohamb2005/gpt2-finetuned-apologies)
- [Emergency Message Model](https://huggingface.co/Sohamb2005/gpt2-finetuned-emergency)

If you want to use these models elsewhere, download them directly from their Hugging Face model pages.
---
## 👨‍💻 Created By

Soham Bacchuwar  
Generative AI Project | May 2025  
Trained, coded, and polished with ❤️

---
