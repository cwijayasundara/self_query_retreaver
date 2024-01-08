import streamlit as st

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings()

persist_directory = './vectorstore'

vector_store = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings)

retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4-1106-preview")

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

st.title("Contact Centre Call Filter Demo:")
st.write("sample Rules!")
st.write("I want to list all the calls where the agent greeted the customer properly in the beginning of the call.")
st.write("I want to list all the calls where the agent did not check at least 2 of last 4 digits of the SSN, "
         "email address, and phone number.")
request = st.text_area('Enter the rule ! ', height=50)

submit = st.button("submit", type="primary")

if submit and request:
    results = chain.invoke(request)
    st.write(results)

