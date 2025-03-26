"""
main.py

Entry point for the Multi-Agent AI system.
Prompts the user for API key before launching UI and agents.
"""

import os
import logging
import streamlit as st
from config import setup_logging
from src.ui import run_ui
from PIL import Image
from src.utils.openai_api import get_supervisor_llm

# ---------------------- Ensure Page Config is Set First ----------------------
st.set_page_config(
    page_title="GenAI Answer Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Setup Logging ----------------------
setup_logging()

# ---------------------- Load User API Key ----------------------
def main():
    """
    Initializes the application and starts the UI.
    Ensures API key is provided before starting any agent processes.
    """
    try:
        logo = Image.open("Images/perrigo-logo.png")
        st.sidebar.image(logo, width=120)
    except Exception:
        st.sidebar.error("Logo image not found.")

    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to continue.")
        return  # Stop execution until user provides an API key

    # Store API key both globally and in session state
    os.environ["OPENAI_API_KEY"] = api_key
    st.session_state["OPENAI_API_KEY"] = api_key

    # Initialize LLM using provided API key
    try:

        st.session_state["llm"] = get_supervisor_llm(api_key)
    except ValueError as e:
        st.error(f"❌ Invalid API Key: {e}")
        return  # Stop execution if API key is invalid

    logging.info("✅ OpenAI API Key set. Starting the Multi-Agent AI System...")

    # Now that the API key is set, start the UI
    try:
        run_ui()
    except Exception as e:
        logging.error(f"Error starting the UI: {e}")
        st.error(f"An error occurred while launching the application: {e}")

if __name__ == "__main__":
    main()
