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

# Define a function to process the uploaded documents
# and update the assistant's files
def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"No ha subido documentos, por favor hágalo para poder seguir.")
    else:
        # Retrieve the file list
        uploaded_files = st.session_state.source_docs
        client_files = client.files.list()

        # Create a set of filenames in client_files for faster lookup
        client_filenames = {file.filename for file in client_files.data}

        uploaded_assitant_files = []
        # Iterate over the uploaded files
        for uploaded_file in uploaded_files:
            # Check if the file already exists in client_files
            if uploaded_file.name in client_filenames:
                # If the file exists, delete it
                file_objects = list(filter(lambda x: x.filename == uploaded_file.name, client_files.data))
                if len(file_objects) > 0:
                    the_file_id = file_objects[0].id
                    delete_status = client.files.delete(the_file_id)

            # Create the new file
            uploaded_assitant_files.append(client.files.create(
                file=uploaded_file,
                purpose='assistants'
            ))

        # Update the assistant's files
        file_ids = [file.id for file in uploaded_assitant_files]
        updated_assistant = client.beta.assistants.update(
            assistant.id,
            tools=[{"type": "retrieval"}],
            file_ids=file_ids,
        )
        st.session_state.assistant = updated_assistant

input_fields()
st.button("Submit Documents", on_click=process_documents)

# Accept user input
if prompt := st.chat_input("Haga su consulta al asistente de Recava..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Reset the assistant if the user has uploaded new documents
    if not st.session_state.assistant:
        st.error("Por favor suba documentos para poder continuar.")
    # Create a new thread if it doesn't exist
    if not st.session_state.thread:
        st.session_state.thread = assistant.beta.threads.create()
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)