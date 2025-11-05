import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_documents(uploaded_file):
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_file_path = f"temp.{file_extension}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Load based on file type
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == "txt":
                loader = TextLoader(temp_file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(temp_file_path)
            else:
                logger.error(f"Unsupported file extension: {file_extension}")
                return []
            
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")
            return docs
        except Exception as e:
            logger.error(f"Error loading file {uploaded_file.name}: {str(e)}")
            return []
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return []

def split_documents(docs):
    if not docs:
        logger.warning("No documents to split")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    if not chunks:
        logger.warning("No chunks to embed")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def setup_rag_chain(vector_store):
    if vector_store is None:
        logger.warning("No vector store provided")
        return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    if os.getenv("GROQ_API_KEY"):
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    else:
        llm = ChatOllama(model="llama3.2", temperature=0)
    template = """You are a helpful assistant. Use the following context to answer the question:
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit App
st.set_page_config(page_title="Smart Document Query", page_icon="📚", layout="wide")

# Font Awesome for icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Custom CSS for stunning UI
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #e0e7ff 0%, #c3e8ff 100%);
            font-family: 'Arial', sans-serif;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput > div > div > input {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #ffffff;
            color: #333;
        }
        .stFileUploader > div > div > div {
            border-radius: 8px;
            padding: 10px;
            background-color: #d1d5db;
            color: #182848;
        }
        .stFileUploader > div > div > div > label {
            color: #182848 !important;
            font-weight: bold;
        }
        .stFileUploader > div > div > div > div > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stFileUploader > div > div > div > div > button:hover {
            background-color: #45a049 !important;
        }
        .stSelectbox > div > div > select {
            border-radius: 8px;
            padding: 10px;
            background-color: #ffffff;
            color: #333;
        }
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #333 !important;
        }
        .dark-theme .stApp {
            background: linear-gradient(135deg, #1c2526 0%, #2e3b3e 100%);
            color: #e0e0e0;
        }
        .dark-theme [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
            color: #e0e0e0;
        }
        .dark-theme .stTextInput > div > div > input {
            background-color: #2e2e2e;
            color: #e0e0e0;
            border: 1px solid #555;
        }
        .dark-theme .stFileUploader > div > div > div {
            background-color: #1a252f;
            color: #ffffff;
        }
        .dark-theme .stFileUploader > div > div > div > label {
            color: #ffffff !important;
            font-weight: bold;
        }
        .dark-theme .stFileUploader > div > div > div > div > button {
            background-color: #4CAF50 !important;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .dark-theme .stFileUploader > div > div > div > div > button:hover {
            background-color: #45a049 !important;
        }
        .dark-theme .stSelectbox > div > div > select {
            background-color: #2e2e2e;
            color: #e0e0e0;
        }
        .dark-theme [data-baseweb="select"] > div {
            background-color: #2e2e2e !important;
            color: #e0e0e0 !important;
        }
        .dark-theme .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .dark-theme .stButton > button:hover {
            background-color: #45a049;
        }
        .dark-theme .stAlert {
            color: #e0e0e0 !important;
            background-color: #2e2e2e !important;
        }
        .dark-theme .stAlert > div {
            color: #e0e0e0 !important;
        }
        .icon {
            margin-right: 10px;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            color: #182848;
            margin-bottom: 20px;
        }
        .dark-theme .title {
            color: #e0e0e0;
        }
        .main-container {
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
        }
        .dark-theme .main-container {
            background-color: rgba(30, 30, 30, 0.9);
        }
        .sidebar-heading {
            font-size: 1.5em;
            font-weight: bold;
            color: #e0e0e0;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Theme Toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

theme = st.session_state.theme
st.markdown(f'<style>.stApp {{ {"background: linear-gradient(135deg, #1c2526 0%, #2e3b3e 100%); color: #e0e0e0;" if theme == "Dark" else ""} }}</style>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-heading"><i class="fas fa-cog icon"></i>Settings</p>', unsafe_allow_html=True)
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], key="theme")

st.markdown(f'<h1 class="title"><i class="fas fa-book-open icon"></i>Smart Document Query</h1>', unsafe_allow_html=True)

# Main content in a styled container
with st.container():
    st.markdown(f'<div class="main-container {"dark-theme" if theme == "Dark" else ""}">', unsafe_allow_html=True)
    
    # File Uploader
    st.sidebar.markdown('<p><i class="fas fa-upload icon"></i>Upload a document</p>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], label_visibility="collapsed")

    # Display file name
    if uploaded_file is not None:
        st.sidebar.markdown(f'<p><i class="fas fa-file icon"></i>Uploaded: {uploaded_file.name}</p>', unsafe_allow_html=True)

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.rag_chain = None

    if st.sidebar.button("Load and Index Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Progress bar
                progress_bar = st.progress(0)
                progress_bar.progress(10)  # Start
                docs = load_documents(uploaded_file)
                progress_bar.progress(50)  # After loading
                if docs:
                    chunks = split_documents(docs)
                    if chunks:
                        st.session_state.vector_store = create_vector_store(chunks)
                        progress_bar.progress(90)  # After vector store
                        if st.session_state.vector_store:
                            st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
                            st.sidebar.success(f"Indexed {uploaded_file.name} successfully!")
                        else:
                            st.sidebar.error("Failed to create vector store. Check the document content.")
                        progress_bar.progress(100)  # Complete
                    else:
                        st.sidebar.error("No valid content extracted from the document.")
                        progress_bar.empty()
                else:
                    st.sidebar.error("Failed to load the document. Ensure it's a valid file.")
                    progress_bar.empty()
        else:
            st.sidebar.error("Please upload a file first.")
            progress_bar.empty()

    st.markdown('<p><i class="fas fa-question-circle icon"></i>Ask a question about the document:</p>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="Type your question here...", label_visibility="collapsed")
    if query and st.session_state.rag_chain:
        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke(query)
                st.markdown(f'<p><strong>Answer:</strong> {response}</p>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    else:
        st.info("Upload and index a document first, then ask a question!")
    
    st.markdown('</div>', unsafe_allow_html=True)