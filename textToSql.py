import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("🧠 Text to SQL")

llm = ChatOpenAI(model="gpt-4o-mini")

schema = """
users(id, name, email, created_at)
orders(id, user_id, amount, created_at)
"""

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