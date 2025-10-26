import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def load_documents(uploaded_file):
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages from the PDF")
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            return []
        finally:
            # Clean up temporary file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
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
st.set_page_config(page_title="RAG AI App", page_icon="🤖", layout="wide")

# Theme Toggle
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { background-color: #1E1E1E; }
        .stApp { background-color: #121212; color: white; }
        .stTextInput > div > div > input { background-color: #2E2E2E; color: white; }
        .stButton > button { background-color: #4CAF50; color: white; }
        .stFileUploader > div > div > div { background-color: #2E2E2E; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { background-color: #FFFFFF; }
        .stApp { background-color: #F0F2F6; color: black; }
        .stTextInput > div > div > input { background-color: #FFFFFF; color: black; }
        .stButton > button { background-color: #4CAF50; color: white; }
        .stFileUploader > div > div > div { background-color: #FFFFFF; color: black; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 RAG AI Question-Answering App")
st.sidebar.header("Settings")

# File Uploader for PDF
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.rag_chain = None

if st.sidebar.button("Load and Index Document"):
    if uploaded_file is not None:
        with st.spinner("Loading and indexing..."):
            docs = load_documents(uploaded_file)
            if docs:
                chunks = split_documents(docs)
                if chunks:
                    st.session_state.vector_store = create_vector_store(chunks)
                    if st.session_state.vector_store:
                        st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
                        st.sidebar.success("Document indexed successfully!")
                    else:
                        st.sidebar.error("Failed to create vector store. Check the PDF content.")
                else:
                    st.sidebar.error("No valid chunks extracted from the PDF.")
            else:
                st.sidebar.error("Failed to load the PDF. Ensure it's a valid, text-based PDF.")
    else:
        st.sidebar.error("Please upload a PDF file first.")

query = st.text_input("Ask a question about the document:")
if query and st.session_state.rag_chain:
    with st.spinner("Generating answer..."):
        try:
            response = st.session_state.rag_chain.invoke(query)
            st.write("**Answer:**", response)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Upload and index a PDF document first, then ask a question!")