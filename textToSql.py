import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from sqlalchemy import create_engine, inspect
import os

# =========================
# DB CONNECTION
# =========================

engine = create_engine(
    "postgresql+psycopg2://postgres:koju%40123@localhost:5432/postgres"
)

# =========================
# OPENAI KEY
# =========================

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================
# STREAMLIT UI
# =========================

st.title("🧠 Text to SQL (RAG Enabled)")

llm = ChatOpenAI(model="gpt-4o-mini")

# =========================
# SCHEMA EXTRACTION (same as yours, but converted to docs)
# =========================

def get_schema_docs(engine):
    inspector = inspect(engine)

    docs = []

    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        cols = [c["name"] for c in columns]

        doc = f"Table: {table}\nColumns: {', '.join(cols)}"
        docs.append(doc)

    return docs

# =========================
# BUILD FAISS INDEX (schema embeddings)
# =========================

@st.cache_resource
def build_vector_db():
    docs = get_schema_docs(engine)

    print("\n================ FULL SCHEMA DOCS ================\n")
    for d in docs:
        print(d)
        print("------------------------------------------------")

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_texts(docs, embeddings)

    return vector_db

vector_db = build_vector_db()

# =========================
# PROMPT
# =========================

template = """
You are a SQL expert.

Convert the question into a valid SQL query.

Rules:
- Return ONLY SQL
- No markdown
- No explanation

Relevant Schema:
{schema}

Question:
{question}
"""

prompt = PromptTemplate.from_template(template)

# =========================
# USER INPUT
# =========================

question = st.text_input("Ask your question")

# =========================
# MAIN FLOW
# =========================

if question:

    # --------------------------------
    # STEP 1: RETRIEVE RELEVANT SCHEMA
    # --------------------------------

    relevant_docs = vector_db.similarity_search(question, k=2)

    schema_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # LOG: retrieved schema
    print("\n================ QUESTION ================\n")
    print(question)

    print("\n================ RETRIEVED SCHEMA ================\n")
    print(schema_context)

    # --------------------------------
    # STEP 2: BUILD PROMPT
    # --------------------------------

    final_prompt = prompt.format(
        schema=schema_context,
        question=question
    )

    # LOG: final prompt
    print("\n================ FINAL PROMPT ================\n")
    print(final_prompt)

    # --------------------------------
    # STEP 3: LLM CALL
    # --------------------------------

    response = llm.invoke(final_prompt)

    sql = response.content.replace("```sql", "").replace("```", "").strip()

    # --------------------------------
    # OUTPUT SQL
    # --------------------------------

    st.subheader("Generated SQL")
    st.code(sql, language="sql")