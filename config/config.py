"""
config.py

Centralized configuration utilities for the generative AI project.
"""

import os
import logging
import openai
import streamlit as st
import faiss
import chromadb
import json
import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage,ToolMessage, AIMessage

from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------------------------------------------------------------
# 2. LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
def setup_logging():
    """
    Sets up logging configuration with file name, function name, and line number.
    Logs messages to the console at INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.info("Logging initialized.")


def display_saved_plot(plot_path: str,):

    """
    Loads and displays a saved plot from the given path in a Streamlit app with a highlighted background.

    Args:
        plot_path (str): Path to the saved plot image.
        bg_color (str): Background color for the image container.
        padding (str): Padding inside the image container.
        border_radius (str): Border radius for rounded corners.
    """

    bg_color: str = "#f0f2f6"
    padding: str = "5px"
    border_radius: str = "10px"
    if os.path.exists(plot_path):
        # Apply styling using markdown with HTML and CSS
        st.markdown(
            f"""
            <style>
                .image-container {{
                    background-color: {bg_color};
                    padding: {padding};
                    border-radius: {border_radius};
                    display: flex;
                    justify-content: center;
                }}
            </style>
            <div class="image-container">
                <img src="data:image/png;base64,{get_base64_image(plot_path)}" style="max-width:100%; height:auto;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"Plot not found at {plot_path}")

def get_base64_image(image_path: str) -> str:
    """
    Converts an image to a base64 string.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Base64-encoded image.
    """
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

