import os
from io import BytesIO
from pathlib import Path
from pymongo import MongoClient
from chromadb import ChromaDB
from bson.binary import Binary
import pickle
import numpy as np
from langchain import FAISS
import streamlit as st
from PyPDF2 import PdfFileReader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")

def extract_text_from_pdf(pdf_path):
    document = []
    try:
        reader = PdfFileReader(BytesIO(pdf_path.read()))
        content = ""
        for page in range(reader.getNumPages()):
            content += reader.getPage(page).extractText()
        document.append(content)
    except Exception as e:
        print(f"Unexpected error with file {pdf_path}: {e}")

    text_splitter = CharacterTextSplitter()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""],
    )
    source_chunks = text_splitter.split_documents(document)
    return source_chunks

# Function to generate embeddings and store them in FAISS
def generate_and_store_embeddings(faiss_index, pdf_paths):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPENAI_API_KEY'])
    for pdf_path in pdf_paths:
        # Extract text from the PDF document
        text = extract_text_from_pdf(pdf_path)
        
        # Generate embedding of the text with OpenAI Embeddings
        document_embedding = embeddings.embed_text(text)
        
        # Store the embedding in FAISS
        faiss_index.add(np.array([document_embedding]))

        # Store the text in MongoDB
        client = MongoClient(os.environ['MONGODB_URI'])
        db = client[os.environ['MONGODB_DB']]
        collection = db[os.environ['MONGODB_COLLECTION']]
        collection.insert_one({"text": text, "vector": Binary(pickle.dumps(document_embedding, protocol=2))})

    return faiss_index

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAIChat(openai_api_key=os.environ['OPENAI_API_KEY']),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def input_fields():
    st.session_state.mongodb_db = st.toggle('Use MongoDB')
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def process_documents():
    faiss_index = FAISS(128)  # Initialize FAISS index with the dimension of your embeddings
    if not st.session_state.source_docs:
        st.warning(f"Please upload the documents.")
    else:
        try:
            retriever = generate_and_store_embeddings(faiss_index, st.session_state.source_docs)
            st.session_state.retriever = retriever
        except Exception as e:
            st.error(f"An error occurred while retrieving embeddings: {e}")


def boot():
    input_fields()
    st.button("Submit Documents", on_click=process_documents)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()