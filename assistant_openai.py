import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Recava Engine - OpenAI Assistant", page_icon=":robot:")
st.title("Recava Chatbot - OpenAI Assistant")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Obtiene el Assitant de OpenAI
assistant = client.beta.assistants.retrieve(st.secrets["OPENAI_ASSITANT_ID"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Suba documentos al corpus", type="pdf", accept_multiple_files=True)

input_fields()
st.button("Submit Documents", on_click=process_documents)

# Accept user input
if prompt := st.chat_input("Haga su consulta al asistente de Recava..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)