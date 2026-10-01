import streamlit as st

st.set_page_config(page_title="Excusify", page_icon="🎭", layout="centered")

import os
import re
import textwrap
import uuid
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import torch
from deep_translator import GoogleTranslator
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPOS = {
    "excuse": "Sohamb2005/gpt2-finetuned-excuses",
    "apology": "Sohamb2005/gpt2-finetuned-apologies",
    "emergency": "Sohamb2005/gpt2-finetuned-emergency",
}
GITHUB_URL = "https://github.com/soham200521/Excusify"

# -------------- SESSION STATE ---------------
for key in ["excuses", "apologies", "emergencies", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []
# latest result of each mode, so it survives reruns (feedback submit, PDF download)
if "results" not in st.session_state:
    st.session_state.results = {}


# -------------- PROMPT FORMAT (identical to training/train_excusify.ipynb) ---------------
def normalize(text):
    """Lower-case, straight quotes, no separator characters, no trailing punctuation."""
    text = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[|:\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(".!?").strip()
    return text.lower()


def excuse_prompt(scenario, urgency, believability, situation):
    return f"{normalize(scenario)} | {normalize(urgency)} | {normalize(believability)} | {normalize(situation)} :"


def apology_prompt(apology_type, situation):
    return f"{normalize(apology_type)} | {normalize(situation)} :"


def emergency_prompt(scenario, details=""):
    details = normalize(details)
    return f"{normalize(scenario)} | {details} :" if details else f"{normalize(scenario)} :"


# -------------- MODELS & GENERATION ---------------
GEN_KWARGS = dict(do_sample=True, top_p=0.92, top_k=50, temperature=0.8,
                  no_repeat_ngram_size=3, repetition_penalty=1.1)
NUM_CANDIDATES = 4

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "you", "your", "our", "but", "not",
    "have", "had", "has", "from", "about", "into", "all", "can", "could", "will", "would", "didn't", "don't",
    "can't", "i'm", "i've", "i'll", "she", "her", "him", "his", "they", "them", "its", "it's", "too", "very",
    "got", "get", "just", "out", "today", "tonight", "yesterday",
}


@st.cache_resource(show_spinner=False)
def load_model(kind):
    repo = MODEL_REPOS[kind]
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo)
    model.eval()
    return tokenizer, model


def clean_completion(text):
    """Keep at most two complete sentences of the model's answer."""
    text = text.split("\n")[0].split(" | ")[0]
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    complete = [s for s in sentences if re.search(r"[.!?]$", s)]
    if complete:
        return " ".join(complete[:2])
    return (text.rstrip(",;:- ") + ".") if text else ""


def generate_candidates(kind, prompt, max_new_tokens=60):
    tokenizer, model = load_model(kind)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, num_return_sequences=NUM_CANDIDATES, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, **GEN_KWARGS,
        )
    start = inputs["input_ids"].shape[1]
    return [clean_completion(tokenizer.decode(seq[start:], skip_special_tokens=True)) for seq in output]


def _stems(text):
    return {w[:5] for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def relevance(candidate, target):
    """How many content words of the user's input the candidate mentions (5-letter stems)."""
    return len(_stems(target) & _stems(candidate))


BLOCKED_WORDS = [
    "gun", "shooter", "suicide", "murder", "kill", "dead", "violence",
    "blood", "assault", "sex", "rape", "alcohol", "drug",
    "overdose", "hang", "stab", "stabbed", "stabbing", "choke", "explosion", "terror", "abuse",
]
# whole words plus common endings, so "deadline", "change" or "skills" are not blocked
_BLOCKED_RE = re.compile(
    r"\b(?:" + "|".join(map(re.escape, BLOCKED_WORDS)) + r")(?:s|es|d|ed|ing|er|ers|y|ly|ist|ists|ism)?\b",
    re.IGNORECASE,
)


def is_appropriate(text, parental_lock=True):
    return (not parental_lock) or _BLOCKED_RE.search(text) is None


def run_generation(kind, prompt, relevance_target, spinner_msg, parental_lock, **gen_kwargs):
    """Samples several candidates and returns the most relevant appropriate one (or None)."""
    try:
        if kind not in st.session_state.get("_loaded", set()):
            with st.spinner(f"Loading the {kind} model (the first time downloads about 500 MB)..."):
                load_model(kind)
            st.session_state.setdefault("_loaded", set()).add(kind)
        with st.spinner(spinner_msg):
            candidates = generate_candidates(kind, prompt, **gen_kwargs)
    except Exception as e:
        st.error(f"⚠️ Couldn't generate text ({MODEL_REPOS[kind]}): {e}")
        return None

    usable = [c for c in candidates if len(c.split()) >= 4]
    if not usable:
        st.error("⚠️ The model didn't return a usable sentence. Please try again.")
        return None
    allowed = [c for c in usable if is_appropriate(c, parental_lock)]
    if not allowed:
        st.error("🚫 Inappropriate content detected and blocked by Parental Control. Please try again.")
        return None
    # highest relevance wins; ties keep the model's sampling order
    return max(enumerate(allowed), key=lambda pair: (relevance(pair[1], relevance_target), -pair[0]))[1]


# -------------- PREVIEWS & EXPORTS ---------------
FONT_CANDIDATES = ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttf"]


@st.cache_resource
def load_font(size):
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    for name in FONT_CANDIDATES:
        for path in (os.path.join(font_dir, name), name):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def image_to_png_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_whatsapp_chat(user_msg, ai_response_msg, user_name="You", sender_name="Recipient", mode="excuse"):
    try:
        width = 700
        bubble_padding = 30
        spacing = 10
        side = 15
        tail_width = 10
        radius = 15
        bg_color = (217, 229, 221)
        user_bubble_color = (220, 248, 198)
        sender_bubble_color = (255, 255, 255)
        text_color = (0, 0, 0)
        font = load_font(16)
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        max_content_width = int(width * 0.75) - 2 * side

        def wrap(text):
            avg_char = font.getlength("A") or 8
            lines = textwrap.wrap(text, width=max(int(max_content_width / avg_char), 20), break_long_words=True)
            wrapped = "\n".join(lines)
            box = measure.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
            return wrapped, box[2] - box[0], box[3] - box[1]

        reply = {"excuse": "Okay, I understand. Take care.", "apology": "Thank you for your apology."}.get(mode, "Okay.")
        messages = []
        for text, is_user in [(f"{user_name}: {user_msg}", True), (f"{user_name}: {ai_response_msg}", True),
                              (f"{sender_name}: {reply}", False)]:
            wrapped, w, h = wrap(text)
            messages.append((wrapped, w, h, is_user))

        height = 2 * side - spacing + sum(h + bubble_padding + spacing for _, _, h, _ in messages)
        img = Image.new("RGB", (width, int(height)), color=bg_color)
        draw = ImageDraw.Draw(img)
        y = side
        for wrapped, w, h, is_user in messages:
            bubble_h = h + bubble_padding
            bubble_w = min(w + 2 * side, max_content_width + 2 * side)
            if is_user:
                x1 = width - bubble_w - side - tail_width
                color = user_bubble_color
                tail = [(width - side - tail_width, y + bubble_h - radius * 1.5), (width - side, y + bubble_h - radius),
                        (width - side - tail_width, y + bubble_h - radius * 0.5)]
            else:
                x1 = side + tail_width
                color = sender_bubble_color
                tail = [(side + tail_width, y + bubble_h - radius * 1.5), (side, y + bubble_h - radius),
                        (side + tail_width, y + bubble_h - radius * 0.5)]
            draw.rounded_rectangle((x1, y, x1 + bubble_w, y + bubble_h), radius=radius, fill=color)
            draw.multiline_text((x1 + side, y + bubble_padding / 2), wrapped, fill=text_color, font=font, spacing=4)
            draw.polygon(tail, fill=color)
            y += bubble_h + spacing
        return img
    except Exception as e:
        print(f"Error in create_whatsapp_chat: {e}")
        return None


def create_sms_chat(message_text, recipient="Contact"):
    try:
        width = 750
        font = load_font(20)
        header_font = load_font(16)
        side_pad, vert_pad, header_space, footer_space = 30, 20, 40, 20
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        wrapped = "\n".join(textwrap.wrap(message_text, width=30, break_long_words=True))
        box = measure.multiline_textbbox((0, 0), wrapped, font=font, spacing=5)
        bubble_w = box[2] - box[0] + 2 * side_pad
        bubble_h = box[3] - box[1] + 2 * vert_pad
        img = Image.new("RGB", (width, max(100, header_space + bubble_h + footer_space)), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((width / 2, 15), f"To: {recipient}", fill=(160, 160, 160), font=header_font, anchor="mt")
        x1 = (width - bubble_w) / 2
        draw.rounded_rectangle((x1, header_space, x1 + bubble_w, header_space + bubble_h), radius=25, fill=(45, 45, 45))
        draw.multiline_text((x1 + side_pad, header_space + vert_pad), wrapped, fill=(230, 230, 230), font=font, spacing=5)
        return img
    except Exception as e:
        print(f"Error in create_sms_chat: {e}")
        return None


def create_pdf(text, header):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=LETTER)
        width, page_height = LETTER
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, page_height - 50, header[:80])
        c.setFont("Helvetica", 12)
        c.drawString(50, page_height - 80, f"Date: {datetime.now().strftime('%B %d, %Y')}")
        c.drawString(50, page_height - 100, "Name: [Your Name Here]")
        text_object = c.beginText(50, page_height - 140)
        text_object.setFont("Helvetica", 12)
        text_object.setLeading(14)
        for paragraph in text.split("\n"):
            for line in simpleSplit(paragraph, "Helvetica", 12, width - 100):
                text_object.textLine(line)
                if text_object.getY() < 100:
                    c.drawText(text_object)
                    c.showPage()
                    text_object = c.beginText(50, page_height - 50)
                    text_object.setFont("Helvetica", 12)
                    text_object.setLeading(14)
        c.drawText(text_object)
        final_y = text_object.getY()
        if final_y < 80:
            c.showPage()
            final_y = page_height - 50
        c.setFont("Helvetica", 12)
        c.drawString(50, final_y - 30, "Signature: _________________________")
        c.save()
        return buffer.getvalue()
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None


def speak_text(text, lang="en"):
    fp = BytesIO()
    gTTS(text=text, lang=lang, slow=False).write_to_fp(fp)
    return fp.getvalue()


def translate_text(text, lang_code, lang_name, notes):
    if lang_code == "en":
        return text
    try:
        with st.spinner(f"Translating to {lang_name}..."):
            return GoogleTranslator(source="auto", target=lang_code).translate(text) or text
    except Exception as e:
        notes.append(f"Translation to {lang_name} failed ({e}). Showing the English version.")
        return text


def build_result(kind, original, lang_name, lang_code, *, heading, box, image, image_caption,
                 pdf_header, pdf_name):
    """Does the slow work once (translation, audio, PDF) and returns a dict kept in session_state."""
    notes = []
    display = translate_text(original, lang_code, lang_name, notes)
    audio = None
    with st.spinner("🎤 Preparing audio..."):
        try:
            audio = speak_text(display, lang_code)
        except Exception as e:
            notes.append(f"Could not generate audio: {e}")
    pdf = create_pdf(original, header=pdf_header)
    if pdf is None:
        notes.append("Could not generate the PDF.")
    if image is None:
        notes.append("Could not generate the preview image.")
    return {
        "id": uuid.uuid4().hex[:8], "kind": kind, "original": original, "display": display, "notes": notes,
        "audio": audio, "heading": heading, "box": box,
        "image": image_to_png_bytes(image) if image is not None else None, "image_caption": image_caption,
        "pdf": pdf, "pdf_name": pdf_name, "feedback_given": False,
    }


# -------------- RESULT & FEEDBACK UI ---------------
FEEDBACK_LABELS = {
    "excuse": ("👍 Liked it?", "⭐ Favorite?", "💯 Rate it (0-10):"),
    "apology": ("👍 Liked it?", "⭐ Favorite?", "💯 Rate it (0-10):"),
    "emergency": ("👍 Effective?", "⭐ Favorite?", "💯 Rate effectiveness (0-10):"),
}


def render_feedback_form(result):
    if result["feedback_given"]:
        st.caption("✅ Feedback recorded for this one. Thank you!")
        return
    liked_label, fav_label, rating_label = FEEDBACK_LABELS[result["kind"]]
    rid = result["id"]
    with st.form(f"feedback_form_{rid}"):
        st.markdown("##### Your feedback")
        cols = st.columns(2)
        liked = cols[0].radio(liked_label, ["Yes", "No"], index=1, horizontal=True, key=f"liked_{rid}")
        favorite = cols[1].radio(fav_label, ["No", "Yes"], index=0, horizontal=True, key=f"fav_{rid}")
        rating = st.slider(rating_label, 0, 10, 5, key=f"rate_{rid}")
        comment = st.text_area("💬 Comments (optional):", key=f"comm_{rid}")
        if st.form_submit_button("Submit Feedback"):
            st.session_state.feedback.append({
                "timestamp": pd.Timestamp.now(tz="UTC"), "item_id": rid, "type": result["kind"],
                "content": result["original"], "liked": liked, "rating": rating, "favorite": favorite,
                "comment": comment,
            })
            result["feedback_given"] = True
            st.toast("Feedback saved! Thank you!", icon="🎉")


def render_result(result):
    st.markdown(result["heading"])
    getattr(st, result["box"])(result["display"])
    for note in result["notes"]:
        st.warning(note)
    if result["audio"]:
        st.audio(result["audio"], format="audio/mp3")
    with st.expander("📎 Preview & export"):
        if result["image"]:
            st.image(result["image"], caption=result["image_caption"])
        if result["pdf"]:
            st.download_button("📄 Download as PDF", data=result["pdf"], file_name=result["pdf_name"],
                               mime="application/pdf", key=f"pdf_{result['id']}")
    render_feedback_form(result)


# -------------- DASHBOARD ---------------
def records_to_df(records, columns):
    return pd.DataFrame(records) if records else pd.DataFrame(columns=columns)


def render_dashboard():
    st.header("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Excuses", len(st.session_state.excuses))
    col2.metric("Apologies", len(st.session_state.apologies))
    col3.metric("Emergency messages", len(st.session_state.emergencies))
    col4.metric("Feedback", len(st.session_state.feedback))
    st.markdown("---")

    df_excuses = records_to_df(st.session_state.excuses, [
        "id", "timestamp", "scenario", "situation", "urgency", "believability", "generated_excuse", "language"])
    df_apologies = records_to_df(st.session_state.apologies, [
        "id", "timestamp", "apology_type", "situation", "generated_apology", "language"])
    df_emergencies = records_to_df(st.session_state.emergencies, [
        "id", "timestamp", "scenario", "details", "generated_emergency", "language"])
    df_feedback = records_to_df(st.session_state.feedback, [
        "timestamp", "item_id", "type", "content", "liked", "rating", "favorite", "comment"])

    if df_feedback.empty:
        st.info("No feedback yet. Generate something and rate it to see the analysis here.")
    else:
        st.subheader("📈 Feedback analysis")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            type_counts = df_feedback["type"].value_counts().rename_axis("type").reset_index(name="count")
            fig = px.pie(type_counts, values="count", names="type", title="Feedback by type", hole=0.3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig)
        with col_chart2:
            st.markdown("##### Liked vs. not liked")
            st.bar_chart(df_feedback["liked"].value_counts())
        st.markdown("---")

        st.subheader("🏆 Excuse leaderboard")
        excuse_feedback = df_feedback[df_feedback["type"] == "excuse"]
        merged = excuse_feedback.merge(df_excuses, left_on="item_id", right_on="id", how="inner",
                                       suffixes=("_fb", "_exc")) if not df_excuses.empty else pd.DataFrame()
        if merged.empty:
            st.info("Rate a few excuses to see the leaderboard.")
        else:
            st.markdown("#### ⭐ Top rated")
            for i, row in enumerate(merged.sort_values("rating", ascending=False).head(3).itertuples(index=False), 1):
                st.markdown(f"**{i}. \"{row.generated_excuse}\"**")
                st.caption(f"Rating {row.rating}/10 · {row.scenario} · for: {row.situation} · "
                           f"liked: {row.liked} · favorite: {row.favorite}")
            st.markdown("#### 📊 Rating distribution")
            st.bar_chart(merged["rating"].value_counts().sort_index())
            sort_cols = ["rating", "timestamp_fb"]
            for title, column, limit in [("#### ❤️ Favorites", "favorite", None), ("#### 👍 Most liked", "liked", 5)]:
                st.markdown(title)
                picked = merged[merged[column] == "Yes"].sort_values(sort_cols, ascending=False)
                if picked.empty:
                    st.info("Nothing here yet.")
                for row in (picked if limit is None else picked.head(limit)).itertuples(index=False):
                    st.markdown(f"- \"{row.generated_excuse}\" (rating {row.rating})")

    st.markdown("---")
    st.subheader("📚 History")
    if not df_excuses.empty:
        with st.expander("📜 Excuses"):
            st.dataframe(df_excuses[["timestamp", "scenario", "situation", "urgency", "believability",
                                     "generated_excuse", "language"]].sort_values("timestamp", ascending=False))
    if not df_apologies.empty:
        with st.expander("💌 Apologies"):
            st.dataframe(df_apologies[["timestamp", "apology_type", "situation", "generated_apology",
                                       "language"]].sort_values("timestamp", ascending=False))
    if not df_emergencies.empty:
        with st.expander("🚨 Emergency messages"):
            st.dataframe(df_emergencies[["timestamp", "scenario", "details", "generated_emergency",
                                         "language"]].sort_values("timestamp", ascending=False))
    if not df_feedback.empty:
        with st.expander("📝 Feedback"):
            st.dataframe(df_feedback.sort_values("timestamp", ascending=False))


def render_footer():
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Built with fine-tuned GPT-2 and Streamlit. "
                "For entertainment purposes only.</p>", unsafe_allow_html=True)


# ================= PAGE =================
st.title("🎭 Excusify")
st.markdown("Excuses, apologies and urgent messages written by GPT-2 models fine-tuned on hand-written datasets.")

st.sidebar.header("⚙️ Settings")
parental_lock = st.sidebar.toggle("Parental Control (filter inappropriate content)", value=True)
st.sidebar.markdown("---")
view = st.sidebar.radio("View", ["🎭 Generator", "📊 Dashboard"], key="view_select")
st.sidebar.markdown("---")
st.sidebar.caption(f"[Source code]({GITHUB_URL}) · Models: "
                   + " · ".join(f"[{k}](https://huggingface.co/{v})" for k, v in MODEL_REPOS.items()))

if view == "📊 Dashboard":
    render_dashboard()
    render_footer()
    st.stop()

LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi",
    "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Japanese": "ja", "Chinese (Simplified)": "zh-CN",
}
LEVELS = ["low", "medium", "high"]

cols_top = st.columns(2)
with cols_top[0]:
    mode = st.selectbox("🎯 Mode:", ["Excuse", "Apology", "Emergency"], key="main_mode_select")
with cols_top[1]:
    lang_name = st.selectbox("🌐 Language:", list(LANGUAGES), key="main_lang_select")
lang_code = LANGUAGES[lang_name]


if mode == "Excuse":
    st.subheader("📝 Excuse details")
    situation = st.text_input("What do you need an excuse for?", "Forgot to submit the report", key="exc_situation")
    cols = st.columns(3)
    with cols[0]:
        scenario = st.selectbox("Context:", ["work", "school", "family", "social"], key="exc_scenario")
    with cols[1]:
        urgency = st.select_slider("Urgency:", LEVELS, value="medium", key="exc_urgency",
                                   help="How serious the reason is: low = minor slip, high = emergency.")
    with cols[2]:
        believability = st.select_slider("Believability:", LEVELS, value="high", key="exc_believability",
                                         help="high = ordinary and believable, low = absurd and funny.")

    if st.button("💡 Generate Excuse", type="primary", key="exc_generate_button"):
        if not situation.strip():
            st.warning("Please describe what you need an excuse for.")
        else:
            excuse = run_generation("excuse", excuse_prompt(scenario, urgency, believability, situation),
                                    situation, "🧠 Thinking of a good excuse...", parental_lock)
            if not excuse:
                st.session_state.results.pop("excuse", None)
            else:
                topic = situation.strip().rstrip(".!?")
                roles = {
                    "work": ("Employee", "Boss", "Hi, I'm really sorry about this."),
                    "school": ("Student", "Teacher", "Good morning, I wanted to explain something."),
                    "family": ("Me", "Family", "Hey, I'm so sorry about this."),
                    "social": ("Me", "Friend", "Hey! So sorry about this."),
                }
                user_name, recipient, opener = roles[scenario]
                result = build_result(
                    "excuse", excuse, lang_name, lang_code, heading="#### ✨ Your excuse:", box="success",
                    image=create_whatsapp_chat(opener, excuse, user_name, recipient, mode="excuse"),
                    image_caption=f"Chat preview: {user_name} → {recipient}",
                    pdf_header=f"Note: {topic}", pdf_name=f"{scenario}_excuse.pdf",
                )
                st.session_state.results["excuse"] = result
                st.session_state.excuses.append({
                    "id": result["id"], "timestamp": pd.Timestamp.now(tz="UTC"), "scenario": scenario,
                    "situation": situation, "urgency": urgency, "believability": believability,
                    "generated_excuse": excuse, "language": lang_name,
                })

    if "excuse" in st.session_state.results:
        render_result(st.session_state.results["excuse"])


elif mode == "Apology":
    st.subheader("💌 Apology details")
    APOLOGY_TYPES = {"emotional": "Emotional (heartfelt)", "professional": "Professional (formal)",
                     "informal": "Informal (casual)"}
    apology_type = st.selectbox("Tone:", list(APOLOGY_TYPES), format_func=APOLOGY_TYPES.get, key="apo_type")
    situation = st.text_input("What are you apologizing for?", "being late to the meeting", key="apo_situation")

    if st.button("🙏 Generate Apology", type="primary", key="apo_generate_button"):
        if not situation.strip():
            st.warning("Please describe what you are apologizing for.")
        else:
            apology = run_generation("apology", apology_prompt(apology_type, situation), situation,
                                     "🖋️ Writing your apology...", parental_lock)
            if not apology:
                st.session_state.results.pop("apology", None)
            else:
                recipient = {"emotional": "Loved one", "professional": "Manager", "informal": "Friend"}[apology_type]
                opener = "Hi, I owe you an apology."
                result = build_result(
                    "apology", apology, lang_name, lang_code, heading="#### ✨ Your apology:", box="info",
                    image=create_whatsapp_chat(opener, apology, "Me", recipient, mode="apology"),
                    image_caption=f"Chat preview: {APOLOGY_TYPES[apology_type]}",
                    pdf_header=f"Apology: {situation.strip().rstrip('.!?')}", pdf_name=f"{apology_type}_apology.pdf",
                )
                st.session_state.results["apology"] = result
                st.session_state.apologies.append({
                    "id": result["id"], "timestamp": pd.Timestamp.now(tz="UTC"), "apology_type": apology_type,
                    "situation": situation, "generated_apology": apology, "language": lang_name,
                })

    if "apology" in st.session_state.results:
        render_result(st.session_state.results["apology"])


elif mode == "Emergency":
    st.subheader("🚨 Emergency message details")
    RECIPIENTS = {
        "work issue": "Manager", "school absence": "Class Teacher", "family matter": "Manager",
        "social event cancellation": "Friends", "car trouble": "Manager", "medical issue": "Manager",
        "stuck somewhere": "Friend", "urgent help needed": "Best Friend",
    }
    scenario = st.selectbox("Scenario:", list(RECIPIENTS), key="em_scenario")
    details = st.text_input("Details (optional, e.g. flat tyre on the highway):", "", key="em_details")

    if st.button("📢 Generate Message", type="primary", key="em_generate_button"):
        message = run_generation("emergency", emergency_prompt(scenario, details), details or scenario,
                                 "📡 Writing an urgent message...", parental_lock, max_new_tokens=70)
        if not message:
            st.session_state.results.pop("emergency", None)
        else:
            result = build_result(
                "emergency", message, lang_name, lang_code, heading="#### ✨ Your message:", box="warning",
                image=create_sms_chat(message, recipient=RECIPIENTS[scenario]),
                image_caption=f"Message preview (to {RECIPIENTS[scenario]})",
                pdf_header=f"Urgent message: {scenario}", pdf_name=f"urgent_{scenario.replace(' ', '_')}.pdf",
            )
            st.session_state.results["emergency"] = result
            st.session_state.emergencies.append({
                "id": result["id"], "timestamp": pd.Timestamp.now(tz="UTC"), "scenario": scenario,
                "details": details, "generated_emergency": message, "language": lang_name,
            })

    if "emergency" in st.session_state.results:
        render_result(st.session_state.results["emergency"])

render_footer()
