import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4

def load_documents(url):
    
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))}
    )
    docs = loader.load()
    return docs
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def setup_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # Get top 4 matches
    
    llm = llm = ChatGroq(model="llama-3.2-3b-preview", temperature=0)  # Free LLaMA model
    
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
# Load environment variables (your API key)
load_dotenv()

# Streamlit App
st.set_page_config(page_title="RAG AI App", page_icon="🤖", layout="wide")

# Theme Toggle in Sidebar
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { background-color: #1E1E1E; }
        .stApp { background-color: #121212; color: white; }
        .stTextInput > div > div > input { background-color: #2E2E2E; color: white; }
        .stButton > button { background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { background-color: #FFFFFF; }
        .stApp { background-color: #F0F2F6; color: black; }
        .stTextInput > div > div > input { background-color: #FFFFFF; color: black; }
        .stButton > button { background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 RAG AI Question-Answering App")
st.sidebar.header("Settings")

# Input for Document URL
url = st.sidebar.text_input("Enter Document URL (e.g., https://lilianweng.github.io/posts/2023-06-23-agent/)", value="https://lilianweng.github.io/posts/2023-06-23-agent/")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.rag_chain = None

if st.sidebar.button("Load and Index Document"):
    with st.spinner("Loading and indexing..."):
        docs = load_documents(url)
        chunks = split_documents(docs)
        st.session_state.vector_store = create_vector_store(chunks)
        st.session_state.rag_chain = setup_rag_chain(st.session_state.vector_store)
    st.sidebar.success("Document indexed successfully!")

# Query Input
query = st.text_input("Ask a question about the document:")
if query and st.session_state.rag_chain:
    with st.spinner("Generating answer..."):
        response = st.session_state.rag_chain.invoke(query)
    st.write("**Answer:**", response)
else:
    st.info("Load a document first, then ask a question!")