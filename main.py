import streamlit as st
from llama_index.core import (
    VectorStoreIndex, 
    ServiceContext, 
    Document, 
    SimpleDirectoryReader
)

import google.generativeai as genai

import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="KMUTT & SCG Chatbot",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Title and header
st.title("KMUTT & SCG Chatbot")

# Configure Google API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about Concrete technology"}]

# Load data
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = GeminiEmbedding(model_name="models/embedding-001", title="this is a document")
        service_context = ServiceContext.from_defaults(llm = Gemini(model="models/gemini-pro"), embed_model=embed_model,)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# User input
if prompt := st.text_input("Your question", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.text_area("User", value=message["content"], disabled=True, key="user_message")
        else:
            st.text_area("Assistant", value=message["content"], disabled=True, key="assistant_message")

# Generate response
if st.button("Send"):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
