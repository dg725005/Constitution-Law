import streamlit as st
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import RetrievalQA

PDF_PATH = "IPC_IndianConstitution.pdf"
FAISS_INDEX_PATH = "faiss_index_store"
BM25_PKL_PATH = "bm25_retriever.pkl"

st.set_page_config(page_title="Law & Constitution Guide", layout="wide")
st.title("⚖️ Law and Constitution Chat Guide")


@st.cache_resource
def get_ensemble_retriever():
    embeddings = OpenAIEmbeddings()
    
    # Check if we need to build the indices
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists('bm25_retriever.pkl'):
        with st.status("Indices not found. Building knowledge base from PDF...", expanded=True) as status:
            st.write("Reading PDF...")
            loader = PyPDFLoader(PDF_PATH)
            pages = loader.load()
            
            st.write("Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pages)
            
            st.write("Generating Vector Embeddings (FAISS)...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            
            st.write("Generating Keyword Index (BM25)...")
            bm25 = BM25Retriever.from_documents(docs)
            with open(BM25_PKL_PATH, "wb") as f:
                pickle.dump(bm25, f)
                
            status.update(label="Index Build Complete!", state="complete", expanded=False)
    
    # LOAD INDICES
    # 1. Load FAISS
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 2. Load BM25
    with open(BM25_PKL_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        bm25_retriever.k = 3
        
    # 3. Fuse them using Reciprocal Rank Fusion (RRF)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )
    return ensemble

# Initialize the retriever
try:
    ensemble_retriever = get_ensemble_retriever()
    
    # Setup the QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        retriever=ensemble_retriever,
        return_source_documents=True
    )

    # --- UI INTERFACE ---
    with st.form(key='chat_form'):
        user_query = st.text_area("Search the IPC or Constitution:", placeholder="Enter query...")
        submit_button = st.form_submit_button(label='Consult Guide')

    if submit_button and user_query:
        with st.spinner("Analyzing legal documents..."):
            result = qa_chain.invoke(user_query)
            
            st.markdown("### 🏛️ Legal Guidance")
            st.write(result['result'])
            
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(result['source_documents']):
                    st.info(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):\n\n{doc.page_content}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Check if your PDF path is correct and your OpenAI API Key is set.")
