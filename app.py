import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
from gtts import gTTS
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from datetime import datetime
import textwrap
import random
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from googletrans import Translator
import base64
import tempfile

# Load GPT2 models from Hugging Face Hub
@st.cache_resource
def load_models():
    excuse_model = AutoModelForCausalLM.from_pretrained("Sohamb2005/gpt2-finetuned-excuses")
    apology_model = AutoModelForCausalLM.from_pretrained("Sohamb2005/gpt2-finetuned-apologies")
    emergency_model = AutoModelForCausalLM.from_pretrained("Sohamb2005/gpt2-finetuned-emergency")
    tokenizer = AutoTokenizer.from_pretrained("Sohamb2005/gpt2-finetuned-excuses")  
    tokenizer.pad_token = tokenizer.eos_token

    excuse_gen = pipeline("text-generation", model=excuse_model, tokenizer=tokenizer)
    apology_gen = pipeline("text-generation", model=apology_model, tokenizer=tokenizer)
    emergency_gen = pipeline("text-generation", model=emergency_model, tokenizer=tokenizer)

    return excuse_gen, apology_gen, emergency_gen, tokenizer

excuse_gen, apology_gen, emergency_gen, tokenizer = load_models()
fake = Faker()
translator = Translator()

language_options = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja", "Russian": "ru"
}

inappropriate_keywords = ["gun", "suicide", "violence", "drugs", "sex", "kill"]

# Utility functions
def speak_text(text, lang='en'):
    tts = gTTS(text, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def generate_text(prompt, generator):
    out = generator(prompt, max_length=40, num_return_sequences=1)[0]['generated_text']
    return out[len(prompt):].split('.')[0] + '.'

def create_pdf(text, header="Official Statement"):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, header)
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%d/%m/%Y')}")
    c.drawString(50, height - 100, "Name: [Redacted]")
    y = height - 140
    lines = simpleSplit(text, "Helvetica", 12, width - 100)
    for line in lines:
        c.drawString(50, y, line)
        y -= 18
    c.drawString(50, y - 30, "Signature: _________________________")
    c.save()
    return temp.name

def is_appropriate(text):
    return not any(word in text.lower() for word in inappropriate_keywords)

def create_chat_image(user_msg, reply_msg, mode="excuse"):
    width, height = 700, 300
    bg_color = (230, 230, 230)
    user_color = (255, 255, 255) if mode == "excuse" else (255, 239, 213)
    reply_color = (0, 132, 255)
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    user_text = textwrap.fill(f"You: {user_msg}", width=40)
    reply_text = textwrap.fill(f"Sender: {reply_msg}", width=40)
    draw.rectangle([50, 50, 650, 100], fill=user_color)
    draw.text((60, 60), user_text, fill=(0, 0, 0), font=font)
    draw.rectangle([50, 150, 650, 200], fill=reply_color)
    draw.text((60, 160), reply_text, fill=(255, 255, 255), font=font)

    return img

def generate_location_log():
    data = []
    base_time = datetime.now()
    for i in range(5):
        timestamp = (base_time - pd.Timedelta(minutes=10*i)).strftime('%Y-%m-%d %H:%M:%S')
        lat, lon = fake.latitude(), fake.longitude()
        address = fake.address().replace('\n', ', ')
        data.append(f"{timestamp} | {lat}, {lon} | {address}")
    return "\n".join(data)

# Streamlit UI
st.set_page_config(page_title="Excuse Generator AI", layout="centered")
st.title("🎭 Intelligent Excuse Generator")

mode = st.selectbox("Choose Mode", ["Excuse", "Apology", "Emergency"])
lang_name = st.selectbox("Language", list(language_options.keys()))
lang_code = language_options[lang_name]

if mode == "Excuse":
    scenario_input = st.text_input("Scenario | Urgency | Believability", "work | high | high")
    reason_input = st.text_input("What do you need an excuse for?", "Late submission")
    if st.button("Generate Excuse"):
        if not scenario_input.strip() or not reason_input.strip():
            st.warning("Please enter all required fields.")
        else:
            prompt = scenario_input.strip() + " :"
            excuse = generate_text(prompt, excuse_gen)
            if not is_appropriate(excuse):
                st.error("🚫 Inappropriate content blocked.")
            else:
                final_excuse = translator.translate(excuse, dest=lang_code).text if lang_code != 'en' else excuse
                st.success(final_excuse)
                st.audio(speak_text(final_excuse, lang_code), format='audio/mp3')
                img = create_chat_image(excuse, "Take care!", mode="excuse")
                st.image(img, caption="WhatsApp-style Chat")
                pdf_path = create_pdf(excuse)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Proof", f, file_name="excuse_proof.pdf")
                st.text(generate_location_log())

elif mode == "Apology":
    apology_type = st.selectbox("Apology Type", ["emotional", "professional"])
    if st.button("Generate Apology"):
        prompt = f"{apology_type} :"
        apology = generate_text(prompt, apology_gen)
        if not is_appropriate(apology):
            st.error("🚫 Inappropriate content blocked.")
        else:
            final_apology = translator.translate(apology, dest=lang_code).text if lang_code != 'en' else apology
            st.success(final_apology)
            st.audio(speak_text(final_apology, lang_code), format='audio/mp3')
            img = create_chat_image(apology, "Thank you for your apology.", mode="apology")
            st.image(img, caption="WhatsApp-style Chat")
            pdf_path = create_pdf(apology, header="Apology Letter")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Letter", f, file_name="apology_letter.pdf")
            st.text(generate_location_log())

elif mode == "Emergency":
    emergency_scenario = st.selectbox("Emergency Scenario", ["work", "school", "family", "social"])
    if st.button("Generate Emergency Message"):
        prompt = f"{emergency_scenario}:"
        message = generate_text(prompt, emergency_gen)
        if not is_appropriate(message):
            st.error("🚫 Inappropriate content blocked.")
        else:
            final_msg = translator.translate(message, dest=lang_code).text if lang_code != 'en' else message
            st.success(final_msg)
            st.audio(speak_text(final_msg, lang_code), format='audio/mp3')
            img = create_chat_image(message, "Alert received.", mode="emergency")
            st.image(img, caption="SMS-style Emergency Alert")
            pdf_path = create_pdf(message, header="Emergency Notification")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Emergency PDF", f, file_name="emergency_alert.pdf")
            st.text(generate_location_log())
