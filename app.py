from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # usefull for chatbot
from langchain_core.output_parsers import (
    StrOutputParser,
)  # You can also create a custom output parser.
import streamlit as str
import os
from dotenv import load_dotenv


# env variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2_API_KEY"] = "true"


# prompt template
prompt = ChatPromptTemplate()
