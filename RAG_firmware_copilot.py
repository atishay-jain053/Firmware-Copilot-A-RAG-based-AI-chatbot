"""
Firmware Copilot — Professional WhatsApp-style chat UI (Refactored for Streamlit)
RAG system for driver code generation from PDF documents.
"""

import os
import sys
import io
import re
import numpy as np
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

BASE_URL = "https://openai.generative.engine.capgemini.com/v1"
KEY = "Put_your_api_key"

os.environ["OPENAI_API_KEY"] = KEY
load_dotenv(override=True)

client = OpenAI(
                base_url=BASE_URL,
                api_key=KEY,
            )           

# Configuration
name = "Firmware Copilot"
#PDF = "document.pdf"

# ---------- Brand tokens ----------
BRAND_GREEN = "#FF5A6B"
BRAND_PURPLE = "#0F6E56"
BRAND_PINK = "#0F1220"
BRAND_GRADIENT = f"linear-gradient(135deg, {BRAND_GREEN} 0%, {BRAND_PURPLE} 50%, {BRAND_PINK} 100%)"


# ---------- Page config ----------
st.set_page_config(
    page_title="Firmware Copilot: Ask Your TRM Anything",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------- Global styling ----------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            /* ---------- Base ---------- */
            html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
                background: #FFFFFF !important;
            }}
            [data-testid="stHeader"] {{
                background: transparent !important;
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 7rem;
                max-width: 820px;
            }}
            * {{
                font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            /* ---------- Sidebar ---------- */
            [data-testid="stSidebar"] {{
                background: #FFFFFF !important;
                border-right: 1px solid #F0F0F4;
            }}
            [data-testid="stSidebar"] .sidebar-brand {{
                display: flex; align-items: center; gap: 12px;
                padding: 8px 4px 18px 4px;
                border-bottom: 1px solid #F2F2F6;
                margin-bottom: 16px;
            }}
            [data-testid="stSidebar"] .sidebar-brand .logo {{
                width: 38px; height: 38px; border-radius: 12px;
                background: {BRAND_GRADIENT};
                box-shadow: 0 6px 18px rgba(138,92,246,0.25);
            }}
            [data-testid="stSidebar"] .sidebar-brand .brand-name {{
                font-weight: 700; font-size: 17px; color: #14141A;
                letter-spacing: -0.01em;
            }}
            [data-testid="stSidebar"] .sidebar-brand .brand-tag {{
                font-size: 11.5px; color: #6E6E80; margin-top: 2px;
            }}
            [data-testid="stSidebar"] .side-section-title {{
                text-transform: uppercase; letter-spacing: 0.08em;
                font-size: 11px; color: #8A8A99; font-weight: 600;
                margin: 14px 0 8px 0;
            }}
            [data-testid="stSidebar"] .side-card {{
                background: #FF5A6B;
                border: 1px solid #EFEFF4;
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 13px; color: #3A3A48;
                line-height: 1.5;
            }}
            [data-testid="stSidebar"] .pill {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(0,230,118,0.10);
                color: #009A52;
                font-size: 11.5px; font-weight: 600;
                margin-right: 6px; margin-top: 6px;
            }}
            [data-testid="stSidebar"] .pill.purple {{
                background: rgba(138,92,246,0.10); color: #6A3FE0;
            }}
            [data-testid="stSidebar"] .pill.pink {{
                background: rgba(255,45,149,0.10); color: #D81B73;
            }}

            /* Sidebar button */
            [data-testid="stSidebar"] .stButton > button {{
                width: 100%;
                border-radius: 12px;
                border: 1px solid #EFEFF4;
                background: #FFFFFF;
                color: #14141A;
                font-weight: 600; font-size: 13px;
                padding: 10px 12px;
                transition: all 0.15s ease;
            }}
            [data-testid="stSidebar"] .stButton > button:hover {{
                border-color: transparent;
                background: {BRAND_GRADIENT};
                color: #FFFFFF;
                box-shadow: 0 8px 22px rgba(138,92,246,0.25);
            }}

            /* ---------- Chat header ---------- */
            .chat-header {{
                position: relative;
                display: flex; align-items: center; gap: 12px;
                padding: 14px 18px;
                border-radius: 18px;
                background: {BRAND_GRADIENT};
                box-shadow: 0 10px 30px rgba(138,92,246,0.22);
                margin-bottom: 18px;
            }}
            .chat-header .avatar {{
                width: 70px; height: 70px;
                border-radius: 50%;
                background: #FFFFFF;
                display: flex; align-items: center; justify-content: center;
                font-weight: 700; color: #15173d; font-size: 11px;
                box-shadow: inset 0 0 0 2px rgba(255,255,255,0.8);
            }}
            .chat-header .meta .name {{
                color: #FFFFFF; font-weight: 700; font-size: 16px;
                letter-spacing: -0.01em;
            }}
            .chat-header .meta .status {{
                display: flex; align-items: center; gap: 6px;
                color: rgba(255,255,255,0.92); font-size: 12px; margin-top: 2px;
            }}
            .chat-header .meta .status .dot {{
                width: 8px; height: 8px; border-radius: 50%;
                background: #FFFFFF;
                box-shadow: 0 0 0 3px rgba(255,255,255,0.35);
            }}
            .chat-header .actions {{
                margin-left: auto;
                display: flex; gap: 8px;
            }}
            .chat-header .actions .icon-btn {{
                width: 34px; height: 34px; border-radius: 10px;
                background: rgba(255,255,255,0.18);
                display: flex; align-items: center; justify-content: center;
                color: #FFFFFF; font-size: 15px;
            }}

            /* ---------- Chat area ---------- */
            .chat-area {{
                background: #FFFFFF;
                padding: 8px 4px 24px 4px;
            }}
            .day-divider {{
                text-align: center;
                margin: 18px 0 14px 0;
            }}
            .day-divider span {{
                background: #F4F4F8;
                color: #6E6E80;
                font-size: 11.5px; font-weight: 600;
                padding: 5px 12px;
                border-radius: 999px;
            }}

            /* Bubble rows */
            .row {{
                display: flex; width: 100%; margin: 6px 0;
            }}
            .row.user {{ justify-content: flex-end; }}
            .row.bot  {{ justify-content: flex-start; }}

            .bubble {{
                max-width: 78%;
                padding: 10px 14px 8px 14px;
                font-size: 14.5px; line-height: 1.5;
                position: relative;
                word-wrap: break-word;
                box-shadow: 0 1px 1px rgba(20,20,26,0.04);
            }}
            .bubble .time {{
                display: block;
                font-size: 10.5px;
                margin-top: 4px;
                opacity: 0.75;
                text-align: right;
            }}

            /* User bubble — brand green, WhatsApp-style right tail */
            .bubble.user {{
                background: {BRAND_GREEN};
                color: #052E16;
                border-radius: 16px 16px 4px 16px;
            }}
            .bubble.user::after {{
                content: "";
                position: absolute;
                right: -6px; bottom: 0;
                width: 12px; height: 14px;
                background: {BRAND_GREEN};
                clip-path: polygon(0 0, 0% 100%, 100% 100%);
                border-bottom-right-radius: 2px;
            }}
            .bubble.user .time {{ color: #064E2A; }}

            /* Bot bubble — soft white card with purple accent rail */
            .bubble.bot {{
                background: #F7F6FC;
                color: #14141A;
                border-radius: 16px 16px 16px 4px;
                border: 1px solid #ECEAF8;
                border-left: 3px solid {BRAND_PURPLE};
            }}
            .bubble.bot::after {{
                content: "";
                position: absolute;
                left: -6px; bottom: 0;
                width: 12px; height: 14px;
                background: #F7F6FC;
                clip-path: polygon(100% 0, 0% 100%, 100% 100%);
                border-bottom-left-radius: 2px;
            }}
            .bubble.bot .time {{ color: #6E6E80; }}

            /* Typing indicator */
            .typing {{
                display: inline-flex; align-items: center; gap: 4px;
                padding: 6px 2px;
            }}
            .typing span {{
                width: 7px; height: 7px; border-radius: 50%;
                background: {BRAND_PURPLE};
                opacity: 0.4;
                animation: blink 1.2s infinite ease-in-out;
            }}
            .typing span:nth-child(2) {{ background: {BRAND_PINK}; animation-delay: 0.15s; }}
            .typing span:nth-child(3) {{ background: {BRAND_GREEN}; animation-delay: 0.3s; }}
            @keyframes blink {{
                0%, 80%, 100% {{ opacity: 0.25; transform: translateY(0); }}
                40%           {{ opacity: 1;    transform: translateY(-2px); }}
            }}

            /* Empty state */
            .empty-state {{
                text-align: center;
                padding: 38px 14px 24px 14px;
                color: #6E6E80;
            }}
            .empty-state .badge {{
                width: 64px; height: 64px; border-radius: 20px;
                background: {BRAND_GRADIENT};
                display: inline-flex; align-items: center; justify-content: center;
                color: #FFFFFF; font-size: 28px;
                box-shadow: 0 14px 36px rgba(138,92,246,0.28);
                margin-bottom: 14px;
            }}
            .empty-state h3 {{
                color: #14141A; margin: 0 0 6px 0; font-size: 20px; font-weight: 700;
                letter-spacing: -0.01em;
            }}
            .empty-state p {{ font-size: 13.5px; margin: 0 0 18px 0; }}
            .suggestions {{
                display: flex; flex-wrap: wrap; gap: 8px;
                justify-content: center;
                margin-top: 6px;
            }}
            .suggestion {{
                font-size: 12.5px;
                color: #3A3A48;
                background: #FAFAFC;
                border: 1px solid #EDEDF3;
                padding: 8px 14px;
                border-radius: 999px;
            }}

            /* ---------- Composer (chat input) ---------- */
            [data-testid="stBottom"],
            [data-testid="stBottom"] > div,
            [data-testid="stBottomBlockContainer"],
            [data-testid="stChatInputContainer"],
            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] > div > div {{
                background: #FFFFFF !important;
            }}
            [data-testid="stBottom"] {{
                border-top: 1px solid #F0F0F4 !important;
                box-shadow: 0 -8px 24px rgba(20,20,26,0.04) !important;
            }}
            [data-testid="stBottomBlockContainer"] {{
                padding-top: 14px !important;
                padding-bottom: 14px !important;
                max-width: 820px;
            }}
            [data-testid="stChatInput"] textarea {{
                background: #FAFAFC !important;
                border: 4px solid #ECECF2 !important;
                border-radius: 16px !important;
                color: #14141A !important;
                padding: 12px 14px !important;
                font-size: 14.5px !important;
            }}
            [data-testid="stChatInput"] textarea:focus {{
                border-color: {BRAND_PURPLE} !important;
                box-shadow: 0 0 0 3px rgba(138,92,246,0.15) !important;
            }}
            [data-testid="stChatInput"] button {{
                background: {BRAND_GRADIENT} !important;
                border-radius: 12px !important;
                color: #FFFFFF !important;
                box-shadow: 0 6px 18px rgba(138,92,246,0.30) !important;
            }}
            [data-testid="stChatInput"] button:hover {{
                filter: brightness(1.05);
            }}

            /* Hide default Streamlit chrome */
            #MainMenu, footer {{ visibility: hidden; }}

            /* Spinner color */
            .stSpinner > div > div {{
                border-top-color: {BRAND_PURPLE} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- UI helpers ----------
def render_header() -> None:
    st.markdown(
        """
        <div class="chat-header">
            <div class="avatar">Firmware<br>Copilot</div>
            <div class="meta">
                <div class="name">Firmware Copilot: Ask Your TRM Anything</div>
                <div class="status"><span class="dot"></span> Online · Replies powered by your Driver Generator Manual</div>
            </div>
            <div class="actions">
                <div class="icon-btn" title="Secure">🔒</div>
                <div class="icon-btn" title="Verified">✓</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> any:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="logo"></div>
                <div>
                    <div class="brand-name">Firmware Copilot</div>
                    <div class="brand-tag">Enterprise · Knowledge Copilot</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="side-card">
                Ask this copilot anything about your Driver Generator Manual to generate code.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-title">Capabilities</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div>
                <span class="pill">Embedded code generation</span>
                <span class="pill purple">Generates .h header file</span>
                <span class="pill pink">Generates .c source file</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section-title">PDF Upload</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        st.markdown('<div class="side-section-title">Session</div>', unsafe_allow_html=True)
        if st.button("🧹  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown(
            """
            <div style="position:absolute; bottom:18px; left:18px; right:18px;
                        font-size:11px; color:#9B9BAC; text-align:center;">
                © Firmware Copilot · Confidential
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        return uploaded_file


def now_hm() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def render_message(role: str, text: str, ts: str) -> None:
    """role ∈ {'user', 'bot'}"""
    safe = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )
    st.markdown(
        f"""
        <div class="row {role}">
            <div class="bubble {role}">
                {safe}
                <span class="time">{ts}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="badge">💬</div>
            <h3>Hi, I'm your Firmware Copilot Assistant</h3>
            <p>Ask me anything about your Driver Generator Manual - I will generate .h and .c file based on your queries.</p>
            <div class="suggestions">
                <div class="suggestion">Generate the code</div>
                <div class="suggestion">Generate</div>
                <div class="suggestion">Generate from manual</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================
# STEP 2 — RAG : Chunking
# ========================
def chunk_text(text, chunk_size=1000, overlap=50):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# =========================
# STEP 3 — RAG : Embedding
# =========================
def get_embeddings(texts):
    """Convert a list of strings into embedding vectors."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


# ========================
# STEP 4 — RAG : Retrieval
# ========================
def cosine_formula(vec_1, vec_2):
    dot = 0
    for a, b in zip(vec_1, vec_2):
        dot += (a * b)

    vec_1_len = 0
    for a in vec_1:
        vec_1_len += (a * a)
    vec_1_len = vec_1_len ** 0.5

    vec_2_len = 0
    for b in vec_2:
        vec_2_len += (b * b)
    vec_2_len = vec_2_len ** 0.5

    similarity = dot / (vec_1_len + vec_2_len)
    return similarity


def retrieve(query, chunks, chunk_embeddings, top_k=3):
    """Find the most relevant chunks for a given question."""
    if chunk_embeddings is None or len(chunks) == 0:
        return []

    query_vec = get_embeddings([query])[0]

    all_score = []
    for chunk_vec in chunk_embeddings:
        score = cosine_formula(query_vec, chunk_vec)
        all_score.append(score)

    indx = list(range(len(all_score)))
    indx.sort(key=lambda i: all_score[i], reverse=True)

    top_k_res = []
    for i in indx[:top_k]:
        top_k_res.append(chunks[i])

    return top_k_res


# ===========================================
# STEP 5 — PROMPT ENGINEERING : System Prompt
# ===========================================
system_prompt = f"""You are acting as {name}.
        You are an embedded systems expert and strict driver generator.
 
        =======================
        STRICT RULES
        =======================
        1. Generate driver code ONLY if the user explicitly requests driver generation.
        2. If input is a greeting or unclear request, respond:
        "Please provide a valid driver generation request."
        3. Use ONLY the provided PDF context.
        4. DO NOT invent registers, macros, or hardware behavior.
        5. If required data is missing, respond:
        "INSUFFICIENT DATA FROM PDF"
 
        =======================
        CONTEXT UNDERSTANDING
        =======================
        You must extract the following:
 
        From USER INPUT:
        - Driver type (I2C, SPI, UART, GPIO, etc.)
        - Device name (if provided)
        - Requested chapter/section (if specified)
 
        From PDF CONTEXT:
        - Identify which chapter/section the context belongs to
        - Extract relevant register and hardware details
 
        =======================
        VALIDATION LOGIC
        =======================
        - If user specifies a chapter:
        → Verify that the provided context matches that chapter
        → If mismatch, respond:
            "CONTEXT DOES NOT MATCH REQUESTED CHAPTER"
 
        - If driver type is not supported by the context:
        → Respond:
            "REQUESTED DRIVER TYPE NOT FOUND IN PDF CONTEXT"
 
        =======================
        FILE NAMING RULE (IMPORTANT)
        =======================
        You MUST dynamically generate file names using:
 
        <device>_<driverType>_ch<chapter>
 
        Examples:
        - eeprom_i2c_ch5.h
        - adc_spi_ch3.c
        - temp_uart_ch2.h
 
        Rules:
        - device → from user input (fallback: "device")
        - driverType → lowercase (i2c/spi/uart/gpio)
        - chapter → number extracted from context/user input
 
        =======================
        OUTPUT FORMAT
        =======================
        ONLY IF ALL VALIDATIONS PASS:
 
        // <generated_file_name>.h
        <code>
 
        // <generated_file_name>.c
        <code>
 
        Do NOT output anything else.

        """


# =======================================================
# STEP 6 — PROMPT ENGINEERING : User Message with Context
# =======================================================
def build_user_message(question, relevant_chunks):
    """
    Wrap the user's question with retrieved context.
    This is the 'Augmented' part of RAG.
    """
    context = " ".join(relevant_chunks)
    return f"""[Relevant info from document]
    {context}
    
    [Question]
    {question}
    
    Answer based only on the provided info above."""


# =============================================
# HELPER: Load and cache PDF processing
# =============================================
@st.cache_resource(show_spinner="Loading document...")
def load_pdf(uploaded_file):
    """Load PDF and extract text into chunks."""
    if uploaded_file is None:
        return "", []
    
    pdf_bytes = io.BytesIO(uploaded_file.read())
    reader = PdfReader(pdf_bytes)
    
    profile = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            profile += text
    
    chunks = chunk_text(profile)
    return profile, chunks


@st.cache_resource(show_spinner="Creating embeddings...")
def prepare_embeddings(chunks):
    """Generate embeddings for all chunks."""
    if not chunks:
        return None
    return get_embeddings(chunks)


# =============================================
# MAIN STREAMLIT APPLICATION
# =============================================
def main() -> None:
    inject_css()
    uploaded_file = render_sidebar()
    render_header()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None

    # Load PDF if uploaded
    if uploaded_file is not None:
        st.session_state.profile, st.session_state.chunks = load_pdf(uploaded_file)
        st.session_state.chunk_embeddings = prepare_embeddings(st.session_state.chunks)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.chunks)} chunks from PDF")

    # Chat area
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="day-divider"><span>{datetime.now().strftime("%A, %d %b %Y")}</span></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            render_empty_state()
        else:
            for m in st.session_state.messages:
                render_message(m["role"], m["text"], m["ts"])

        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type a message…")
    if user_input:
        if not st.session_state.chunks:
            st.error("⚠️ Please upload a PDF first!")
        else:
            # Add user message
            st.session_state.messages.append(
                {"role": "user", "text": user_input, "ts": now_hm()}
            )
            render_message("user", user_input, now_hm())

            # Show typing indicator
            typing_slot = st.empty()
            typing_slot.markdown(
                """
                <div class="row bot">
                    <div class="bubble bot">
                        <div class="typing"><span></span><span></span><span></span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                # RAG: Retrieve relevant chunks
                relevant = retrieve(
                    user_input,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    top_k=3
                )
                
                # Build augmented message with context
                augmented_message = build_user_message(user_input, relevant)

                # Prepare messages for API
                history = [{"role": "system", "content": system_prompt}]
                
                # Add conversation history (excluding current user message)
                for msg in st.session_state.messages[:-1]:
                    history.append({"role": msg["role"], "content": msg["text"]})
                
                # Add current augmented message
                history.append({"role": "user", "content": augmented_message})

                # Call OpenAI
                response = client.chat.completions.create(
                    model="openai.gpt-5-nano",
                    messages=history,
                    temperature=0.3,
                    max_tokens=300,
                )

                answer = response.choices[0].message.content

            except Exception as exc:
                answer = f"Sorry, I hit an error while answering: {exc}"

            typing_slot.empty()

            # Add bot response
            st.session_state.messages.append(
                {"role": "assistant", "text": str(answer), "ts": now_hm()}
            )
            render_message("assistant", str(answer), now_hm())

            # Handle file generation
            output_dir = os.path.join(os.getcwd(), "generated_drivers")
            os.makedirs(output_dir, exist_ok=True)

            invalid_markers = [
                "Please provide a valid driver generation request",
                "INSUFFICIENT DATA",
                "CONTEXT DOES NOT MATCH",
                "REQUESTED DRIVER TYPE NOT FOUND"
            ]

            if not any(marker in answer for marker in invalid_markers):
                pattern = r"//\s*(\S+\.(h|c))\n([\s\S]*?)(?=(//\s*\S+\.(h|c))|\Z)"
                matches = re.findall(pattern, answer)

                for match in matches:
                    filename = match[0]
                    code = match[2]

                    driver_folder = filename.split(".")[0]
                    driver_dir = os.path.join(output_dir, driver_folder)
                    os.makedirs(driver_dir, exist_ok=True)

                    file_path = os.path.join(driver_dir, filename)

                    with open(file_path, "w") as f:
                        f.write(code.strip())

                    st.success(f"✅ File created: {file_path}")

            st.rerun()


if __name__ == "__main__":
    main()
