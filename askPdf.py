import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 1. Load PDF from Downloads
pdf_path = os.path.expanduser("~/Downloads/Sujan-Koju-Resume.pdf")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings + vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# 4. Ask question
query = input("Ask something about the PDF: ")

# 5. Similarity search
matched_docs = db.similarity_search(query)

# 6. LLM + QA chain
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")

# 7. Get answer
response = chain.run(input_documents=matched_docs, question=query)

print("\nAnswer:", response)