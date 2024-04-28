import streamlit as st
import os
from dotenv import load_dotenv


# ENV VARIABLES
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
