import streamlit as st
from openai import OpenAI
import time
import uuid

st.set_page_config(page_title="Recava Engine - OpenAI Assistant", page_icon=":robot:")
st.title("Recava Chatbot - OpenAI Assistant")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Obtiene el Assitant de OpenAI
st.session_state.assistant = client.beta.assistants.retrieve(st.secrets["OPENAI_ASSITANT_ID"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state: # Used to identify each session
    st.session_state.session_id = str(uuid.uuid4())

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
            st.session_state.assistant.id,
            tools=[{"type": "retrieval"}],
            file_ids=file_ids,
        )
        st.session_state.assistant = updated_assistant

input_fields()
st.button("Submit Documents", on_click=process_documents)

# Create a new thread for this session
st.session_state.thread = client.beta.threads.create(
    metadata={
        'session_id': st.session_state.session_id,
    }
)

# If the run is completed, display the messages
if hasattr(st.session_state.run, 'status') and st.session_state.run.status == "completed":
    # Retrieve the list of messages
    st.session_state.messages = client.beta.threads.messages.list(
        thread_id=st.session_state.thread.id
    )

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    with st.chat_message('user'):
        st.write(prompt)

    # Add message to the thread
    st.session_state.messages = client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=prompt
    )

# Do a run to process the messages in the thread
    st.session_state.run = client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id,
    )
    if st.session_state.retry_error < 3:
        time.sleep(1) # Wait 1 second before checking run status
        st.rerun()

# Check if 'run' object has 'status' attribute
if hasattr(st.session_state.run, 'status'):
    # Handle the 'running' status
    if st.session_state.run.status == "running":
        with st.chat_message('assistant'):
            st.write("Thinking ......")
        if st.session_state.retry_error < 3:
            time.sleep(1)  # Short delay to prevent immediate rerun, adjust as needed
            st.rerun()

    # Handle the 'failed' status
    elif st.session_state.run.status == "failed":
        st.session_state.retry_error += 1
        with st.chat_message('assistant'):
            if st.session_state.retry_error < 3:
                st.write("Run failed, retrying ......")
                time.sleep(3)  # Longer delay before retrying
                st.rerun()
            else:
                st.error("FAILED: The OpenAI API is currently processing too many requests. Please try again later ......")

    # Handle any status that is not 'completed'
    elif st.session_state.run.status != "completed":
        # Attempt to retrieve the run again, possibly redundant if there's no other status but 'running' or 'failed'
        st.session_state.run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state.thread.id,
            run_id=st.session_state.run.id,
        )
        if st.session_state.retry_error < 3:
            time.sleep(3)
            st.rerun()