import streamlit as st

st.set_page_config(page_title="Excuse Generator AI", layout="centered")

import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from gtts import gTTS
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from datetime import datetime
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from deep_translator import GoogleTranslator
import tempfile

# -------------- SESSION STATE INIT ---------------
if "excuses" not in st.session_state:
    st.session_state.excuses = []
if "apologies" not in st.session_state:
    st.session_state.apologies = []
if "emergencies" not in st.session_state:
    st.session_state.emergencies = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

fake = Faker()

# -------------- UTILITY FUNCTIONS ---------------
def speak_text(text, lang='en'):
    tts = gTTS(text, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def is_appropriate(text, parental_lock=True):
    inappropriate_keywords = [
        "gun", "shooter", "suicide", "murder", "kill", "dead", "violence",
        "blood", "assault", "sex", "rape", "alcohol", "drugs",
        "overdose", "hang", "stab", "choke", "explosion", "terror", "abuse"
    ]
    if not parental_lock:
        return True
    return not any(word in text.lower() for word in inappropriate_keywords)

def create_whatsapp_chat(user_msg, sender_msg, user_name="You", sender_name="Sender", mode="excuse"):
    width, height = 700, 300
    bg_color = (230, 230, 230)
    user_bubble_color = (255, 255, 255) if mode == "excuse" else (255, 239, 213)
    sender_bubble_color = (0, 132, 255)
    user_text_color = (0, 0, 0)
    sender_text_color = (255, 255, 255)
    font = ImageFont.load_default()
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    user_text = textwrap.fill(f"{user_name}: {user_msg}", width=40)
    user_bbox = draw.multiline_textbbox((0, 0), user_text, font=font)
    user_w, user_h = user_bbox[2] - user_bbox[0], user_bbox[3] - user_bbox[1]
    user_bubble = (width - user_w - 50, 30, width - 20, 30 + user_h + 20)
    draw.rounded_rectangle(user_bubble, fill=user_bubble_color, radius=20, outline=(200,200,200))
    draw.multiline_text((user_bubble[0]+15, user_bubble[1]+10), user_text, fill=user_text_color, font=font)
    sender_text = textwrap.fill(f"{sender_name}: {sender_msg}", width=40)
    sender_bbox = draw.multiline_textbbox((0, 0), sender_text, font=font)
    sender_w, sender_h = sender_bbox[2] - sender_bbox[0], sender_bbox[3] - sender_bbox[1]
    sender_bubble = (20, height - sender_h - 60, 20 + sender_w + 30, height - 40)
    draw.rounded_rectangle(sender_bubble, fill=sender_bubble_color, radius=20)
    draw.multiline_text((sender_bubble[0]+15, sender_bubble[1]+10), sender_text, fill=sender_text_color, font=font)
    return img

def create_sms_chat(message_text, sender_name="XX-NDMAEW"):
    width, height = 750, 180
    background = (0, 0, 0)
    bubble_color = (50, 50, 50)
    text_color = (255, 255, 255)
    font = ImageFont.load_default()
    img = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(img)
    wrapped_text = textwrap.fill(message_text, width=50)
    bubble_x, bubble_y = 20, 60
    bubble_w, bubble_h = draw.multiline_textbbox((0, 0), wrapped_text, font=font)[2:]
    bubble_box = (bubble_x, bubble_y, bubble_x + bubble_w + 30, bubble_y + bubble_h + 30)
    draw.rounded_rectangle(bubble_box, radius=20, fill=bubble_color)
    draw.multiline_text((bubble_x + 15, bubble_y + 15), wrapped_text, fill=text_color, font=font)
    draw.text((width // 2 - 50, 15), sender_name, fill=(180, 180, 180), font=font)
    return img

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

def generate_location_log(num_entries=5):
    data = []
    base_time = datetime.now()
    for i in range(num_entries):
        timestamp = (base_time.replace(second=0, microsecond=0) - pd.Timedelta(minutes=10*i)).strftime('%Y-%m-%d %H:%M:%S')
        lat, lon = fake.latitude(), fake.longitude()
        address = fake.address().replace('\n', ', ')
        data.append(f"{timestamp} | {lat}, {lon} | {address}")
    return "\n".join(data)

# -------------- LOAD MODELS ---------------
@st.cache_resource
def load_models():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    excuse_model = GPT2LMHeadModel.from_pretrained("Sohamb2005/gpt2-finetuned-excuses")
    apology_model = GPT2LMHeadModel.from_pretrained("Sohamb2005/gpt2-finetuned-apologies")
    emergency_model = GPT2LMHeadModel.from_pretrained("Sohamb2005/gpt2-finetuned-emergency")
    excuse_gen = pipeline("text-generation", model=excuse_model, tokenizer=tokenizer)
    apology_gen = pipeline("text-generation", model=apology_model, tokenizer=tokenizer)
    emergency_gen = pipeline("text-generation", model=emergency_model, tokenizer=tokenizer)
    return excuse_gen, apology_gen, emergency_gen, tokenizer

excuse_gen, apology_gen, emergency_gen, tokenizer = load_models()

# -------------- APP LAYOUT ---------------
st.title("🎭 Intelligent Excuse Generator")

# Sidebar: Parental Control and Dashboard
st.sidebar.header("Settings")
parental_lock = st.sidebar.toggle("Parental Control (filter inappropriate content)", value=True)

if st.sidebar.button("Show Dashboard"):
    st.header("📊 Dashboard")

    # Show counts
    total_excuses = len(st.session_state.excuses)
    total_apologies = len(st.session_state.apologies)
    total_emergencies = len(st.session_state.emergencies)
    st.write(f"**Total Excuses:** {total_excuses}")
    st.write(f"**Total Apologies:** {total_apologies}")
    st.write(f"**Total Emergencies:** {total_emergencies}")

    # Prepare DataFrames
    df_excuses = pd.DataFrame(st.session_state.excuses) if st.session_state.excuses else pd.DataFrame()
    df_feedback = pd.DataFrame(st.session_state.feedback) if st.session_state.feedback else pd.DataFrame()

    # Merge feedback with excuses for ranking (if both exist)
    if not df_excuses.empty and not df_feedback.empty:
        df_merged = df_feedback[df_feedback['type'] == 'excuse'].merge(
            df_excuses, left_on='content', right_on='generated_excuse', how='left'
        )
    else:
        df_merged = pd.DataFrame()

    # Ranking: Top-rated and most liked/favorited excuses
    if not df_merged.empty:
        st.subheader("🏆 Top Excuses (by Rating)")
        top_rated = df_merged.sort_values("rating", ascending=False).head(3)
        for i, row in top_rated.iterrows():
            st.markdown(f"**{row['content']}**  \nRating: {row['rating']} | Likes: {row['liked']} | Favorite: {row['favorite']}")

        st.subheader("⭐ Favorite Excuses")
        favorites = df_merged[df_merged['favorite'] == 'Yes']
        if not favorites.empty:
            for i, row in favorites.iterrows():
                st.markdown(f"**{row['content']}**  \nRating: {row['rating']}")
        else:
            st.info("No favorite excuses yet.")

        st.subheader("👍 Most Liked Excuses")
        likes = df_merged[df_merged['liked'] == 'Yes']
        if not likes.empty:
            for i, row in likes.iterrows():
                st.markdown(f"**{row['content']}**  \nRating: {row['rating']}")
        else:
            st.info("No liked excuses yet.")

        st.subheader("📈 Ratings Distribution")
        st.bar_chart(df_merged['rating'].value_counts().sort_index())
    else:
        st.info("No feedback or excuses to rank yet.")

    # Show all feedback table
    if not df_feedback.empty:
        st.subheader("All Feedback")
        st.dataframe(df_feedback)
    else:
        st.info("No feedback yet.")

    # Show all excuses table
    if not df_excuses.empty:
        st.subheader("All Generated Excuses")
        st.dataframe(df_excuses)
    else:
        st.info("No excuses yet.")

    st.stop()

# -------------- MAIN APP ---------------
language_options = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Italian": "it", "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja", "Russian": "ru"
}

mode = st.selectbox("Choose Mode", ["Excuse", "Apology", "Emergency"])
lang_name = st.selectbox("Language", list(language_options.keys()))
lang_code = language_options[lang_name]

def generate_text(prompt, generator):
    out = generator(prompt, max_length=40, num_return_sequences=1)[0]['generated_text']
    return out[len(prompt):].split('.')[0] + '.'

if mode == "Excuse":
    scenario_input = st.text_input("Scenario | Urgency | Believability", "work | high | high")
    reason_input = st.text_input("What do you need an excuse for?", "Late submission")
    if st.button("Generate Excuse"):
        if not scenario_input.strip() or not reason_input.strip():
            st.warning("Please enter all required fields.")
        else:
            prompt = scenario_input.strip() + " :"
            excuse = generate_text(prompt, excuse_gen)
            if not is_appropriate(excuse, parental_lock):
                st.error("🚫 Inappropriate content blocked.")
            else:
                final_excuse = GoogleTranslator(source='auto', target=lang_code).translate(excuse) if lang_code != 'en' else excuse
                st.success(final_excuse)
                st.audio(speak_text(final_excuse, lang_code), format='audio/mp3')
                img = create_whatsapp_chat(excuse, "Take care!", mode="excuse")
                st.image(img, caption="WhatsApp-style Chat")
                pdf_path = create_pdf(excuse)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Proof", f, file_name="excuse_proof.pdf")
                st.text(generate_location_log())
                # Save to session state
                st.session_state.excuses.append({
                    'timestamp': pd.Timestamp.now(),
                    'scenario': scenario_input.split('|')[0].strip().lower(),
                    'urgency': scenario_input.split('|')[1].strip().lower(),
                    'believability': scenario_input.split('|')[2].strip().lower(),
                    'what_excuse_for': reason_input,
                    'generated_excuse': excuse
                })
                # Feedback widgets in a form
                with st.form("excuse_feedback_form"):
                    st.subheader("Feedback")
                    liked = st.radio("Did you like this excuse?", ["Yes", "No"])
                    favorite = st.radio("Mark as Favorite?", ["Yes", "No"])
                    rating = st.slider("Rate this excuse (0-10):", 0, 10, 5)
                    comment = st.text_area("Comments (optional):")
                    submitted = st.form_submit_button("Save Feedback")
                    if submitted:
                        st.session_state.feedback.append({
                            'timestamp': pd.Timestamp.now(),
                            'type': 'excuse',
                            'content': excuse,
                            'liked': liked,
                            'rating': rating,
                            'favorite': favorite,
                            'comment': comment
                        })
                        st.success("Feedback saved!")

elif mode == "Apology":
    apology_type = st.selectbox("Apology Type", ["emotional", "professional"])
    if st.button("Generate Apology"):
        prompt = f"{apology_type} :"
        apology = generate_text(prompt, apology_gen)
        if not is_appropriate(apology, parental_lock):
            st.error("🚫 Inappropriate content blocked.")
        else:
            final_apology = GoogleTranslator(source='auto', target=lang_code).translate(apology) if lang_code != 'en' else apology
            st.success(final_apology)
            st.audio(speak_text(final_apology, lang_code), format='audio/mp3')
            img = create_whatsapp_chat(apology, "Thank you for your apology.", mode="apology")
            st.image(img, caption="WhatsApp-style Chat")
            pdf_path = create_pdf(apology, header="Apology Letter")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Letter", f, file_name="apology_letter.pdf")
            st.text(generate_location_log())
            # Save to session state
            st.session_state.apologies.append({
                'timestamp': pd.Timestamp.now(),
                'apology_type': apology_type,
                'generated_apology': apology
            })
            # Feedback widgets in a form
            with st.form("apology_feedback_form"):
                st.subheader("Feedback")
                liked = st.radio("Did you like this apology?", ["Yes", "No"])
                favorite = st.radio("Mark as Favorite?", ["Yes", "No"])
                rating = st.slider("Rate this apology (0-10):", 0, 10, 5)
                comment = st.text_area("Comments (optional):")
                submitted = st.form_submit_button("Save Feedback")
                if submitted:
                    st.session_state.feedback.append({
                        'timestamp': pd.Timestamp.now(),
                        'type': 'apology',
                        'content': apology,
                        'liked': liked,
                        'rating': rating,
                        'favorite': favorite,
                        'comment': comment
                    })
                    st.success("Feedback saved!")

elif mode == "Emergency":
    emergency_scenario = st.selectbox("Emergency Scenario", ["work", "school", "family", "social"])
    if st.button("Generate Emergency Message"):
        prompt = f"{emergency_scenario}:"
        message = generate_text(prompt, emergency_gen)
        if not is_appropriate(message, parental_lock):
            st.error("🚫 Inappropriate content blocked.")
        else:
            final_msg = GoogleTranslator(source='auto', target=lang_code).translate(message) if lang_code != 'en' else message
            st.success(final_msg)
            st.audio(speak_text(final_msg, lang_code), format='audio/mp3')
            img = create_sms_chat(message)
            st.image(img, caption="SMS-style Emergency Alert")
            pdf_path = create_pdf(message, header="Emergency Notification")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Emergency PDF", f, file_name="emergency_alert.pdf")
            st.text(generate_location_log())
            # Save to session state
            st.session_state.emergencies.append({
                'timestamp': pd.Timestamp.now(),
                'scenario': emergency_scenario,
                'generated_emergency': message
            })
            # Feedback widgets in a form
            with st.form("emergency_feedback_form"):
                st.subheader("Feedback")
                liked = st.radio("Did you like this emergency message?", ["Yes", "No"])
                favorite = st.radio("Mark as Favorite?", ["Yes", "No"])
                rating = st.slider("Rate this emergency message (0-10):", 0, 10, 5)
                comment = st.text_area("Comments (optional):")
                submitted = st.form_submit_button("Save Feedback")
                if submitted:
                    st.session_state.feedback.append({
                        'timestamp': pd.Timestamp.now(),
                        'type': 'emergency',
                        'content': message,
                        'liked': liked,
                        'rating': rating,
                        'favorite': favorite,
                        'comment': comment
                    })
                    st.success("Feedback saved!")
