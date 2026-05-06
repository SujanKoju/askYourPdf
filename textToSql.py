import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sqlalchemy import create_engine, inspect, text
import os


engine = create_engine(
    "postgresql+psycopg2://postgres:koju%40123@localhost:5432/postgres"
)

def get_schema(engine):
    inspector = inspect(engine)

    schema_lines = []
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        cols = [c["name"] for c in columns]
        schema_lines.append(f"{table}({', '.join(cols)})")

    return "\n".join(schema_lines)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("🧠 Text to SQL")

llm = ChatOpenAI(model="gpt-4o-mini")

schema = get_schema(engine)
print(schema)

template = """
You are a SQL expert.

Convert the question into a valid executable SQL query.

Rules:
- Return ONLY the SQL query
- Do NOT include ``` or markdown
- Do NOT include explanations
- Output must be directly executable

Schema:
{schema}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

question = st.text_input("Ask your question")

if question:
    final_prompt = prompt.format(schema=schema, question=question)
    response = llm.invoke(final_prompt)

    st.code(response.content, language="sql")