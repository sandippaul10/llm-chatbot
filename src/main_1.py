import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader

# Load the environment variables
load_dotenv()


def load_documents(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    doc_chunks = text_splitter.split_text(documents)
    vectorstore = FAISS.from_texts(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )
    return chain

# Frontend part

st.set_page_config(
    page_title="Chat with Doc",
    layout="centered"
)

st.title("Chat with Doc")

# Initialize the chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF file", type=["pdf"])

if uploaded_file:

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_documents(uploaded_file))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your query")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
