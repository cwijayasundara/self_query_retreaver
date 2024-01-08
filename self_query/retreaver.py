import streamlit as st

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

load_dotenv()

vectorstore = Chroma(persist_directory="./vectorstore",
                     embedding_function=OpenAIEmbeddings())
# create self-querying retriever

metadata_field_info = [
    AttributeInfo(
        name="call_id",
        description="The unique identifier for the call",
        type="string",
    ),
    AttributeInfo(
        name="call_date_time",
        description="The date and time of the call",
        type="string",
    ),
    AttributeInfo(
        name="call_duration",
        description="The duration of the call",
        type="string",
    ),
    AttributeInfo(
        name="agent",
        description="The agent that handled the call",
        type="string"
    ),
]
document_content_description = "The transcript of the call between the agent and the customer"

llm = ChatOpenAI(temperature=0,
                 model="gpt-4-1106-preview")

prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()

query_constructor = (prompt
                     | llm
                     | output_parser)

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
)

st.title("Contact Centre Call Filter Demo:")
st.write("Sample Rules!")
st.write("I want to list all the calls where the agent greeted the customer properly in the beginning of the call.")
st.write("I want to list all the calls where the agent did not check at least 2 of last 4 digits of the SSN, "
         "email address, and phone number.")
request = st.text_area('Enter the rule ! ', height=50)
submit = st.button("submit", type="primary")

if submit and request:
    results = retriever.invoke(request)
    if results is None:
        st.write("No results found")
    else:
        for res in results:
            st.write(res.page_content)
            st.write("\n" + "-" * 20 + "\n")
