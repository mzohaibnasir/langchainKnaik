from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # usefull for chatbot
from langchain_core.output_parsers import (
    StrOutputParser,
)  # You can also create a custom output parser.
import streamlit as st
import os
from dotenv import load_dotenv


# env variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# langchain
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Please respond to queries"),
        ("user", "Question:{question}"),
    ]
)

# streamlit framework

st.title("Langchain Demo")
input_text = st.text_input("search the topic you want to know about")


# openAI llm
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()  # responsible for getting output

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
