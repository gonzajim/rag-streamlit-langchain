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

def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"No ha subido documentos, por favor hágalo para poder seguir.")
    else:
        # Retrieve the file list
        uploaded_files = st.session_state.source_docs
        client_files = client.files.list()
        for file in client_files.data:
            st.write(f"File: {file.filename}")
            # Find the file by name
            filename_to_find = file.filename
            the_file_id = None
            file_objects = list(filter(lambda x: x.filename == filename_to_find, uploaded_files.data))
            if len(file_objects) > 0:
                the_file_id = file_objects[0].id
            # Si el archivo ya existe, lo borramos
            if the_file_id:
                delete_status = client.files.delete(the_file_id)
                st.warning(f"Se ha borrado el archivo {filename_to_find} con status: {delete_status}")
                # Creamos el archivo actualizado
                file = client.files.create(
                    file=file,
                    purpose='assistants'
                )
        # Recupero de nuevo la lista de ficheros para asignarla al asistente
        client_files = client.files.list()
        updated_assistant = client.beta.assistants.update(
            assistant.id,
            tools=[{"type": "retrieval"}],
            file_ids=client_files,
            )
        
        try:
            assistant.corpus.upload(st.session_state.source_docs)
        except Exception as e:
            st.error(f"Error al subir documentos: {e}")

input_fields()
st.button("Submit Documents", on_click=process_documents)

# Accept user input
if prompt := st.chat_input("Haga su consulta al asistente de Recava..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)