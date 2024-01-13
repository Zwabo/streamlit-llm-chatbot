import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

def get_text():
  input_text = st.chat_input("Hello, how are you?")
  return input_text

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    
    return db


if 'chain' not in st.session_state:
  load_dotenv()
  openai_api_key = os.getenv("OPENAI_API_KEY")

  llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
  chain = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())
  st.session_state['chain'] = chain
else:
  chain = st.session_state.chain

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain RAG Chatbot", page_icon=":robot:")
st.header("LangChain RAG Chatbot")
pdf = st.file_uploader('Upload a PDF file to ask questions about it', type='pdf')

if pdf is not None:
  pdf_reader = PdfReader(pdf)
  text = ""
  for page in pdf_reader.pages:
      text += page.extract_text()
  
  db = process_text(text)

  if "messages" not in st.session_state:
      st.session_state.messages = []

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

  user_input = get_text()

  if user_input:
    docs = db.similarity_search(user_input)

    prompt = f"""
    Please use the content of the following [PDF] to answer my question. If you don't know, please say you don't know, and the answer should be concise."
    [PDF]: {docs}
    Please answer this question in conjunction with the above PDF: {user_input}
"""

    output = chain.predict(input=prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      full_response = ""
      # Simulate stream of response with milliseconds delay
      for chunk in output.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
      message_placeholder.markdown(full_response)
    # Store assistant message to storage
    st.session_state.messages.append({"role": "assistant", "content": full_response})