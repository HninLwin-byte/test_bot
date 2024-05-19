# import streamlit as st
# import openai
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini
# from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core import (
#     VectorStoreIndex, 
#     ServiceContext, 
#     Document, 
#     SimpleDirectoryReader
# )
# from streamlit_extras.app_logo import add_logo
# import google.generativeai as genai

# import dotenv,os
# dotenv.load_dotenv()


# st.set_page_config(page_title="UjjwalDeepXIXC", page_icon=r"scg_logo.jpg", layout="centered", initial_sidebar_state="auto", menu_items=None)
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #e6f7ff;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# col1, col2 = st.columns([5, 1])

# # Display the logo in the first column
# with col1:
#     st.image('scg_logo.jpg', width=120)
    
    
#     st.header(":violet[SCG & KMUTT Chat]Bot",divider='rainbow', help = "This bot is designed by Ujjwal Deep to address all of your questions hehe")



# # col1, col2 = st.columns([1, 3])

# # # Display the logo in the first column
# # with col1:
# #     st.image('scg_logo.jpg', width=100)

# # # Display the text in the second column
# # with col2:
# st.subheader("Hello! There, How can I help you Today-  :)")
   

# st.caption(":violet[what a] :orange[good day] :violet[to share what SCG is offering right now!]")


# # st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# # st.title("KMUTT & SCG Chatbot")

# # add sidebar buttons


# # add sidebar filters
# # st.sidebar.slider("Slider", 0, 100, 50)
# # st.sidebar.date_input("Date Input")
# # openai.api_key = st.secrets.openai_key
# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)


# if "messages" not in st.session_state.keys(): # Initialize the chat messages history
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about Concrete technology"}
#     ]

# @st.cache_resource(show_spinner=False)
# def load_data():
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert o$
#         # index = VectorStoreIndex.from_documents(docs)
#         # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
#         embed_model = GeminiEmbedding(
#             model_name="models/embedding-001", title="this is a document"
#             )
#         service_context = ServiceContext.from_defaults(llm = Gemini(model="models/gemini-pro"), embed_model=embed_model,)
#         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
#         return index

# index = load_data()

# if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#         st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

# for message in st.session_state.messages: # Display the prior chat messages
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # for message in st.session_state.messages:
# #     with st.container():
# #         if message["role"] == "assistant":
# #             st.image('scg_logo.jpg', width=30)
# #             st.write("ChatBot:", message["content"])
# #         else:
# #             st.write("You:", message["content"])

# # If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.write(response.response)
#             message = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message) # Add response to message history

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import glob

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")
    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")

    # Define path to PDF files
    pdf_path = "./data/*"  # Update this path to the folder containing your PDFs
    pdf_docs = glob.glob(pdf_path)

    # Process PDFs automatically upon app launch
    if not st.session_state.get('processed', False):
        with st.spinner("Processing PDF files..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.session_state['processed'] = True
            st.success("PDF files processed")

    # Chat input and response handling
    if prompt := st.chat_input("Ask a question based on the PDF content:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        response = user_input(prompt)
        if response:
            full_response = ''.join(response['output_text'])
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.write(full_response)

if __name__ == "__main__":
    main()





