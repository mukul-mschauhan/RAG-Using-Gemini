import os
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.schema import Document
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

# ✅ Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

# ✅ Configure Gemini & Embedding Model
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Streamlit UI
st.header("📘📋🎯RAG Assistant: :blue[HF Embeddings + Gemini LLM]")
st.subheader("📢Your Intelligent Document✍️Assistant!!")

# ✅ File Uploader
uploaded_file = st.file_uploader(
    "Upload your PDF Document",
    type="pdf",
    help="Select a PDF document for analysis"
)

if uploaded_file:
    raw_text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    if raw_text.strip():
        # ✅ Prepare document chunks
        document = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([document])

        # ✅ Generate embeddings and create FAISS vector store
        texts = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(texts, embedding_model)
        retriever = vector_db.as_retriever()

        st.success("✅ Document processed successfully! Ask your question below.")

        # ✅ User Query Input
        user_query = st.text_input("Enter your question:")

        if user_query:
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Analyzing the Document..."):
                relevant_docs = retriever.get_relevant_documents(user_query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # ✅ Prompt for Gemini
                prompt = f"""
                You are an expert assistant. Use the context below to answer the query.
                If unsure, say 'I don't know.'

                Context:
                {context}

                Query:
                {user_query}

                Answer:
                """

                # ✅ Generate Answer from Gemini
                response = gemini_model.generate_content(prompt)
                st.markdown("### 📌 Answer:")
                st.write(response.text)
    else:
        st.warning("⚠️ No text could be extracted from the PDF. Please upload a readable document.")