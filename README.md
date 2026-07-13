Professional WhatsApp-style Streamlit UI for RAG-based driver code generation

This project provides a Streamlit application that implements a Retrieval-Augmented Generation (RAG) workflow for generating C/.h driver files from PDF device manuals.

**Features**
Upload a PDF (Driver Generator Manual) and the app will chunk and embed its text.
Ask questions in a chat-style UI; the system retrieves relevant context and answers.
When a valid driver generation response is returned, the app extracts and saves generated .h and .c files to generated_drivers/.

**Prerequisites**
Python 3.9+
An OpenAI-compatible API endpoint and API key

**Installation**
1.Clone or download this repository.
2.Create and activate a virtual environment:
  python -m venv .venv
  # Windows
    .venv\Scripts\activate
  # macOS / Linux
    source .venv/bin/activate
3.Install dependencies:
  pip install -r requirements.txt
  
**Configuration**
The app expects an API key to be provided via environment variable OPENAI_API_KEY or the file .env (the project uses python-dotenv).

Create a .env file in the project root with:

OPENAI_API_KEY=your_api_key_here

If you are using a custom base URL for an enterprise/generative endpoint, edit BASE_URL and KEY inside RAG_firmware_copilot.py or set the environment variables appropriately.

**Running the app**
Start the Streamlit app:
  streamlit run RAG_firmware_copilot.py

Open the URL printed by Streamlit in your browser. Upload a PDF using the sidebar, then ask the assistant to generate driver code.

Generated files will be written under generated_drivers/<generated_file_name_without_ext>/<file>.

**Files of interest**
RAG_firmware_copilot.py — main Streamlit app

**Notes**
1.The system prompt enforces strict validation; if required data is missing from the PDF, the assistant will reply with an explicit error message (e.g., INSUFFICIENT DATA FROM PDF).
2.This repository contains a placeholder API key in the source; do not commit real secrets.
