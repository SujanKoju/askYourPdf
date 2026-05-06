import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain

import tempfile
import os

# -----------------------------
# API KEY
# -----------------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📄 Chat with your PDF")

# -----------------------------
# SESSION STATE (IMPORTANT)
# -----------------------------
if "qa" not in st.session_state:
    st.session_state.qa = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# UPLOAD PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.qa is None:
    with st.spinner("Processing PDF..."):

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text / Chunking
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        # Embeddings + Vector DB
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

        # Conversational RAG chain
        st.session_state.qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            retriever=db.as_retriever()
        )

    st.success("PDF ready! Start chatting below 👇")

# -----------------------------
# SHOW CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
if prompt := st.chat_input("Ask something about your PDF..."):

    if st.session_state.qa is None:
        st.warning("Please upload a PDF first.")
    else:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        result = st.session_state.qa({
            "question": prompt,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.chat_history.append((prompt, answer))

        with st.chat_message("assistant"):
            st.markdown(answer)