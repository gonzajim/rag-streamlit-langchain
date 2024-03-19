import os
from io import BytesIO
from pathlib import Path
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import numpy as np
import langchain_community.vectorstores as faiss
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

def load_documents(doc_files):
    documents = []
    for doc_file in doc_files:
        try:
            reader = PdfFileReader(BytesIO(doc_file.read()))
            content = ""
            for page in range(reader.getNumPages()):
                content += reader.getPage(page).extractText()
            documents.append(content)
        except Exception as e:
            print(f"Unexpected error with file {doc_file}: {e}")
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter()
    texts = []
    for document in documents:
        texts.extend(text_splitter.split_text(document))
    return texts

@st.cache(allow_output_mutation=True)
def embeddings_on_local_vectordb(texts):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    source_chunks = text_splitter.split_documents(texts)
    # Get embeddings for each document
    doc_embeddings = [embeddings.embed(doc) for doc in source_chunks]

    # Stack embeddings into a matrix
    embeddings_matrix = np.vstack(doc_embeddings)

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])

    # Add vectors to the index
    index.add(embeddings_matrix)

    return index

def embeddings_on_mongodb(texts):
    client = MongoClient(os.environ['MONGODB_URI'])
    db = client[os.environ['MONGODB_DB']]
    collection = db[os.environ['MONGODB_COLLECTION']]
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    for text in texts:
        vector = embeddings.embed(text)
        collection.insert_one({"text": text, "vector": Binary(pickle.dumps(vector, protocol=2))})
    
    retriever = None  # You need to implement a retriever that can fetch and search vectors from MongoDB
    return retriever

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
    if not st.session_state.source_docs:
        st.warning(f"Please upload the documents.")
    else:
        try:
            documents = load_documents(st.session_state.source_docs)
        except Exception as e:
            st.error(f"An error occurred while loading documents: {e}")

        try:
            texts = split_documents(documents)
        except Exception as e:
            st.error(f"An error occurred while splitting documents: {e}")

        try:
            if not st.session_state.mongodb_db:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
            else:
                st.session_state.retriever = embeddings_on_mongodb(texts)
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