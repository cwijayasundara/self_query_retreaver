import pandas as pd
import json
import streamlit as st

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)

load_dotenv()

call_data = pd.read_csv("data/call_details.csv")

columns = call_data.columns
columns = columns.drop("call_transcript")
print(columns)

model = ChatOpenAI(model="gpt-4")

attribute_info = model.predict(
    "Below is the column heading information about calls came to a contact centre. "
    "Return a JSON list with an entry for each column. Each entry should have "
    '{"name": "column name", "description": "column description", "type": "column data type"}'
    f"\n\n{columns}\n\nJSON:\n"
)

attribute_info = json.loads(attribute_info)
print(attribute_info)

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

from langchain.schema import Document

docs = []

# loop through the call_data
for call in call_data.fillna("").iterrows():
    call_transcript = call[1]["call_transcript"]
    metadata = call[1].drop("call_transcript").to_dict()
    doc = Document(
        page_content=call_transcript,
        metadata=metadata
    )
    docs.append(doc)

# print the contents of the first document
print(docs[0].page_content)
print(docs[0].metadata)

vecstore = Chroma.from_documents(docs, embeddings)

#  retriever
doc_contents = "call transcript"
prompt = get_query_constructor_prompt(doc_contents, attribute_info)
print(prompt.format(query="{query}"))

chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-4-1106-preview", temperature=0), doc_contents, attribute_info
)

from langchain.retrievers import SelfQueryRetriever

retriever = SelfQueryRetriever(
    query_constructor=chain, vectorstore=vecstore, verbose=True
)

st.title("Contact Centre Call Filter Demo:")
st.write("sample Rules!")
st.write("I want to list all the calls where the agent greeted the customer properly in the beginning of the call.")
st.write("I want to list all the calls where the agent did not check at least 2 of last 4 digits of the SSN, "
         "email address, and phone number.")
request = st.text_area('Enter the rule ! ', height=50)
submit = st.button("submit", type="primary")

if submit and request:
    results = retriever.get_relevant_documents(request)
    for res in results:
        st.write(res.page_content)
        st.write("\n" + "-" * 20 + "\n")
