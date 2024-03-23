import os
from io import BytesIO
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from pymongo import MongoClient
import llm_helper

st.set_page_config(page_title="RAG Recava UCLM")
st.title("Retrieval Augmented Generation - RECAVA - UCLM")

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in pdf.pages)
    return text

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

def save_embeddings_to_mongo(embedded_docs, embeddings, index_name="uclm_corpus"):
    # Initialize MongoDB python client
    client = MongoClient(os.environ['MONGODB_URI'])
    collection = client[os.environ['MONGODB_DB']][os.environ['MONGODB_COLLECTION']]

    # Insert the documents in MongoDB Atlas with their embedding
    docsearch = MongoDBAtlasVectorSearch.from_texts(
        embedded_docs, embeddings, collection=collection, index_name=index_name
    )
    return docsearch

def get_embeddings_from_mongo():
    # Initialize MongoDB python client
    client = MongoClient(os.environ['MONGODB_URI'])
    collection = client[os.environ['MONGODB_DB']][os.environ['MONGODB_COLLECTION']]

    # Load embeddings from MongoDB
    embeddings = []
    documents = collection.find()
    for doc in documents:
        st.write(f"Collection recuperada: {doc}")
    

    return documents

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
    st.session_state.source_docs = st.file_uploader(label="Suba documentos al corpus", type="pdf", accept_multiple_files=True)

def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"No ha subido documentos, por favor hágalo para poder seguir.")
    else:
        try:
            all_chunks = []
            for uploaded_file in st.session_state.source_docs:
                # Leo un documento y extraigo su texto
                text = extract_text_from_pdf(uploaded_file)
                chunks = get_text_chunks(text)
                # Guardo los chunks en una lista con todos los libros
                all_chunks.extend(chunks)
            
            # Genero los embeddings de los chunks
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPENAI_API_KEY'])
            db = FAISS.from_texts(all_chunks, embeddings)
            st.write(f"Indice de FAISS: {db.index.ntotal}")

            st.session_state.retriever = db.as_retriever()
            st.session_state.index = db.index

            #Guardo los embeddings en MongoDB
            #docsearch = save_embeddings_to_mongo(all_chunks, embeddings, index_name="uclm_corpus")

            # Recupero los embeddings de MongoDB
            #index = get_embeddings_from_mongo()
            #st.write(f"Indice de FAISS: {docsearch.search_by_vector(embeddings[1], top_k=10)}")
        except Exception as e:
            st.error(f"An error occurred while retrieving embeddings: {e}")

input_fields()
st.button("Submit Documents", on_click=process_documents)

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Introduzca su pregunta...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        # Add a placeholder for the loading message
        loading_message = st.empty()
        loading_message.text("Estamos procesando su pregunta...")

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()
        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")
        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs
        
        # get the chain with the retrieval callback
        custom_chain = get_rag_chain(retrieval_cb=retrieval_cb, vectorstore=db.index)
    
        if "messages" in st.session_state:
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = ""
        for response in custom_chain.stream(
            {"input": prompt, "chat_history": chat_history}
        ):
            if "output" in response:
                full_response += response["output"]
            else:
                full_response += response.content

            message_placeholder.markdown(full_response + "▌")
            update_retrieval_status()

        retrieval_status.update(state="complete")
        message_placeholder.markdown(full_response)

        # Once the response is ready, clear the loading message
        loading_message.empty()

        # add the full response to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})