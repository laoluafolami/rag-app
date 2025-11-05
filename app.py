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

# ----------------------------------------------------------------------
# Logging & env
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ----------------------------------------------------------------------
# Document Functions
# ----------------------------------------------------------------------
def load_documents(uploaded_file):
    if uploaded_file is None:
        return []
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_file_path = f"temp.{file_extension}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
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
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def split_documents(docs):
    if not docs:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    if not chunks:
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

# ----------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------
st.set_page_config(page_title="Smart Document Query", page_icon="book", layout="wide")

# Font Awesome CDN
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

# Custom CSS with Icons
st.markdown(
    """
    <style>
        .stApp { 
            background: linear-gradient(135deg, #e0e7ff 0%, #c3e8ff 100%); 
            font-family: 'Segoe UI', sans-serif; 
        }
        [data-testid="stSidebar"] { 
            background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .stButton > button { 
            background: linear-gradient(135deg, #4CAF50, #45a049); 
            color: white; 
            border-radius: 12px; 
            padding: 12px 24px; 
            font-weight: bold; 
            font-size: 1.1em;
            border: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }
        .stButton > button:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 6px 12px rgba(0,0,0,0.3); 
        }
        .stTextInput > div > div > input { 
            border-radius: 12px; 
            padding: 12px; 
            border: 2px solid #ccc; 
            background-color: #ffffff; 
            color: #333; 
            font-size: 1.1em;
        }
        .stFileUploader > div > div > div { 
            border-radius: 12px; 
            padding: 12px; 
            background-color: #e3f2fd; 
            border: 2px dashed #42a5f5;
        }
        .stFileUploader > div > div > div > label { 
            color: #1565c0 !important; 
            font-weight: bold; 
            font-size: 1.1em;
        }
        .stFileUploader > div > div > div > div > button { 
            background: linear-gradient(135deg, #42a5f5, #1e88e5) !important; 
            color: white !important; 
            border-radius: 12px; 
            padding: 10px 20px; 
            font-weight: bold; 
        }
        .dark-theme .stApp { 
            background: linear-gradient(135deg, #1c2526 0%, #2e3b3e 100%); 
            color: #e0e0e0; 
        }
        .dark-theme [data-testid="stSidebar"] { 
            background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%); 
        }
        .dark-theme .stTextInput > div > div > input { 
            background-color: #2e2e2e; 
            color: #e0e0e0; 
            border: 2px solid #555; 
        }
        .dark-theme .stFileUploader > div > div > div { 
            background-color: #1a252f; 
            border: 2px dashed #4a90e2;
        }
        .dark-theme .stFileUploader > div > div > div > label { 
            color: #4a90e2 !important; 
        }
        .icon { margin-right: 10px; font-size: 1.3em; }
        .title { 
            text-align: center; 
            font-size: 2.8em; 
            color: #182848; 
            margin-bottom: 20px; 
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .dark-theme .title { color: #e0e0e0; }
        .main-container { 
            padding: 30px; 
            border-radius: 16px; 
            background-color: rgba(255, 255, 255, 0.95); 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); 
            max-width: 900px; 
            margin: auto; 
            backdrop-filter: blur(10px);
        }
        .dark-theme .main-container { 
            background-color: rgba(30, 30, 30, 0.95); 
        }
        .sidebar-heading { 
            font-size: 1.6em; 
            font-weight: bold; 
            color: #e0e0e0; 
            margin-bottom: 15px; 
            display: flex; 
            align-items: center;
        }
        .sidebar-heading i { margin-right: 12px; font-size: 1.4em; }
    </style>
    """,
    unsafe_allow_html=True
)

# Theme Toggle with Icons
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

theme = st.session_state.theme
theme_icon = "moon" if theme == "Light" else "sun"
st.sidebar.markdown(f'<p class="sidebar-heading"><i class="fas fa-{theme_icon} icon"></i>Theme</p>', unsafe_allow_html=True)
new_theme = st.sidebar.selectbox("", ["Light", "Dark"], index=0 if theme == "Light" else 1, key="theme_selector")
if new_theme != theme:
    st.session_state.theme = new_theme
    st.rerun()

# Apply theme background
st.markdown(
    f'<style>.stApp {{ {"background: linear-gradient(135deg, #1c2526 0%, #2e3b3e 100%); color: #e0e0e0;" if st.session_state.theme == "Dark" else ""} }}</style>',
    unsafe_allow_html=True
)

# App Title
st.markdown(f'<h1 class="title"><i class="fas fa-book-open icon"></i>Smart Document Query</h1>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Main Container
# ----------------------------------------------------------------------
with st.container():
    st.markdown(f'<div class="main-container {"dark-theme" if st.session_state.theme == "Dark" else ""}">', unsafe_allow_html=True)

    # File Uploader
    st.sidebar.markdown('<p class="sidebar-heading"><i class="fas fa-cloud-upload-alt icon"></i>Upload Document</p>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.sidebar.markdown(f'<p><i class="fas fa-file-alt icon"></i><strong>Uploaded:</strong> {uploaded_file.name}</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.rag_chain = None

    # ---- Load and Index Document ---------------------------------------
    if st.sidebar.button("Load & Index Document"):
        progress_bar = st.progress(0)

        if uploaded_file is None:
            st.sidebar.error("Please upload a file first.")
            progress_bar.empty()
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
        else:
            with st.spinner("Processing your document..."):
                progress_bar.progress(10)
                docs = load_documents(uploaded_file)
                progress_bar.progress(50)
                if not docs:
                    st.sidebar.error("Failed to load document.")
                    progress_bar.empty()
                    st.session_state.vector_store = None
                    st.session_state.rag_chain = None
                else:
                    chunks = split_documents(docs)
                    if not chunks:
                        st.sidebar.error("No content extracted.")
                        progress_bar.empty()
                        st.session_state.vector_store = None
                        st.session_state.rag_chain = None
                    else:
                        vector_store = create_vector_store(chunks)
                        progress_bar.progress(90)
                        if vector_store is None:
                            st.sidebar.error("Failed to create vector store.")
                            progress_bar.empty()
                            st.session_state.vector_store = None
                            st.session_state.rag_chain = None
                        else:
                            st.session_state.vector_store = vector_store
                            st.session_state.rag_chain = setup_rag_chain(vector_store)
                            progress_bar.progress(100)
                            st.sidebar.success(f"Indexed **{uploaded_file.name}** successfully!")

    # ---- Ask Question --------------------------------------------------
    st.markdown('<p style="font-size:1.2em; font-weight:bold; color:#182848;"><i class="fas fa-message-question icon"></i>Ask a question about the document:</p>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="e.g., What is the main topic?", label_visibility="collapsed")

    if query:
        if st.session_state.rag_chain is None:
            st.warning("No document indexed yet. Upload a file and click **Load & Index Document** first.")
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state.rag_chain.invoke(query)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.info("Upload a document → Click **Load & Index** → Ask your question")

    st.markdown('</div>', unsafe_allow_html=True)