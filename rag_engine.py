import os
from io import BytesIO
import numpy as np
from langchain import FAISS
import streamlit as st
from PyPDF2 import PdfReader
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


st.set_page_config(page_title="RAG Recava UCLM")
st.title("Retrieval Augmented Generation - RECAVA - UCLM")

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in pdf.pages)
    return text

def extract_pages_from_pdf(uploaded_file):
    # Crea un objeto PdfFileReader
    pdf = PdfReader(uploaded_file)
    # Extrae el texto de cada página
    pages = [pdf.pages[i].extract_text() for i in range(len(pdf.pages))]
    st.write(f"Páginas que tiene el libro: {pages}")
    return pages

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return chunks

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Suba documentos al corpus", type="pdf", accept_multiple_files=True)

def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"No ha subido documentos, por favor hágalo para poder seguir.")
    else:
        try:
            all_chunks = []
            all_docs = []
            for uploaded_file in st.session_state.source_docs:
                # Leo un documento y extraigo su texto
                text = extract_text_from_pdf(uploaded_file)
                chunks = get_text_chunks(text)
                # Guardo los chunks en una lista con todos los libros
                all_chunks.extend(chunks)
                # Guardo las páginas del documento en una lista
                all_docs.append(extract_pages_from_pdf(uploaded_file))
            
            # Genero los embeddings de los chunks
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPENAI_API_KEY'])
            db = FAISS.from_documents(all_docs, embeddings)
            st.write(f"Indice de FAISS: {db.index.ntotal}")
        except Exception as e:
            st.error(f"An error occurred while retrieving embeddings: {e}")


def boot():
    input_fields()
    st.button("Subir documentos", on_click=process_documents)
    if st.session_state.source_docs:
        st.write(f"Documentos subidos: {len(st.session_state.source_docs)}")

if __name__ == '__main__':
    boot()