import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import (
    VectorStoreIndex, 
    ServiceContext, 
    Document, 
    SimpleDirectoryReader
)
from streamlit_extras.app_logo import add_logo
import google.generativeai as genai
from genai import GenerativeModel 
from llama_index.llms.huggingface import HuggingFaceLLM

# Create a new service context based on Hugging Face's Transformers



import dotenv,os
dotenv.load_dotenv()






st.set_page_config(page_title="SCG&KMUTT chatbot", page_icon=r"scg_logo.jpg", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([5, 1])

# Display the logo in the first column
with col1:
    st.image('scg_logo.jpg', width=120)
    
    
    st.header(":violet[SCG & KMUTT Chat]Bot",divider='rainbow', help = "This bot is designed by Ujjwal Deep to address all of your questions hehe")




st.subheader("Hello! There, How can I help you Today-  :)")
   

st.caption(":violet[what a] :orange[good day] :violet[to share what SCG is offering right now!]")



GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Concrete technology"}
    ]

@st.cache_resource(show_spinner=False)

def load_data():
    try:
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            
            embed_model = GeminiEmbedding(
                model_name="models/embedding-001", title="this is a document"
            )

            genai.GenerativeModel(model_name='gemini-pro')

            llm = Gemini(model_name="models/gemini-pro", api_key = "AIzaSyB02OFMsZeDsGGnJhTYm7_p8Z3Eci6ErbY")

            
           

           
            service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
            
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            
            return index
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
index = load_data()
# def load_data():
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         # # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert o$
#         # # index = VectorStoreIndex.from_documents(docs)
#         # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
#         embed_model = GeminiEmbedding(
#             model_name="models/embedding-001", title="this is a document"
#              )
# #         llm = HuggingFaceLLM(
# #     context_window=4096,
# #     max_new_tokens=256,
# #     generate_kwargs={"temperature": 0, "do_sample": False},
# #     system_prompt=system_prompt,
# #     query_wrapper_prompt=query_wrapper_prompt,
# #     tokenizer_name="mistralai/Mistral-7B-v0.1",
# #     model_name="mistralai/Mistral-7B-v0.1",
# #     device_map="auto",
# #     tokenizer_kwargs={"max_length": 4096},
# #     # uncomment this if using CUDA to reduce memory usage
# #     model_kwargs={
# #         "torch_dtype": torch.float16, 
# #         "llm_int8_enable_fp32_cpu_offload": True,
# #         "bnb_4bit_quant_type": 'nf4',
# #         "bnb_4bit_use_double_quant":True,
# #         "bnb_4bit_compute_dtype":torch.bfloat16,
# #         "load_in_4bit": True}
# # )
#         model = genai.GenerativeModel("gemini-pro")
#         service_context = ServiceContext.from_defaults(llm =model, embed_model=embed_model,)
#         # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
#         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
#         return index

# index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# for message in st.session_state.messages:
#     with st.container():
#         if message["role"] == "assistant":
#             st.image('scg_logo.jpg', width=30)
#             st.write("ChatBot:", message["content"])
#         else:
#             st.write("You:", message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history



