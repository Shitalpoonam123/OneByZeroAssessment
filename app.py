# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from htmlTemplates import bot_template, user_template, css
import os

load_dotenv()

#set environment variables for openai
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

def get_pdf_text(pdf_files):
    """
    Extract text from multiple PDF files.
    
    Args:
        pdf_files (list): List of uploaded PDF file objects.
    
    Returns:
        str: Concatenated text from all PDF files.
    """
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    """
    Split the input text into smaller chunks.
    
    Args:
        text (str): Input text to be split.
    
    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Create a vector store from text chunks using Azure OpenAI embeddings.
    
    Args:
        text_chunks (list): List of text chunks.
    
    Returns:
        FAISS: FAISS vector store object.
    """
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_MODEL")
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    """
    Create a conversational retrieval chain using Azure OpenAI.
    
    Args:
        vector_store (FAISS): FAISS vector store object.
    
    Returns:
        ConversationalRetrievalChain: Conversation chain object.
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_MODEL"),  
        api_version=os.getenv("OPENAI_API_VERSION")
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(question):
    """
    Process user input and display the chat history.
    
    Args:
        question (str): User's input question.
    """
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit app.
    """
    load_dotenv()
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Build your own Chat Bot with Your own pdf:books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        if st.session_state.conversation is not None:
            handle_user_input(question)
        else:
            st.write("Please upload and process your PDFs first.")

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            if pdf_files:
                with st.spinner("Processing your PDFs..."):
                    # Get PDF Text
                    raw_text = get_pdf_text(pdf_files)

                    # Get Text Chunks
                    text_chunks = get_chunk_text(raw_text)

                    # Create Vector Store
                    vector_store = get_vector_store(text_chunks)
                    st.write("DONE")

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
            else:
                st.write("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
