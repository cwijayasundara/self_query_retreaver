from dotenv import load_dotenv
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator

load_dotenv()

transcript_1_positive = """agent(emily rogers) hello this is Emily from the banking support team how may I assist you 
today? customer(michael clark) hi emily I need help with an unrecognized charge on my account agent(emily rogers) I 
understand your concern michael could you please provide me with your email address and the last 4 digits of your SSN 
for verification purposes customer(michael clark) absolutely it's mike.clark@inbox.com and the last digits are 4321 
agent(emily rogers) thank you, I have raised a conflict for this transaction and we’ll keep you updated on the 
progress of this. (michael clark) thanks goodbye"""

transcript_2_no_sec_check = """agent(Sarah Bennett) Good morning, how may I assist you today? customer(James Smith) Hello, 
I've noticed a wrongful charge on my credit card. agent(Sarah Bennett) I'm sorry to hear that, James. Which charge 
are you referring to, and what date was it on? customer(James Smith) It's a charge for $200 on the 1st of April. 
agent(Sarah Bennett) Alright, I see it. I will initiate a dispute claim for you immediately. You'll get an email 
confirmation shortly. customer(James Smith) That's awesome, thanks! agent(Sarah Bennett) My pleasure, 
is there anything else I can assist with? customer(James Smith) No, that's everything. agent(Sarah Bennett) Great, 
have a wonderful day, James. customer(James Smith) You too, goodbye."""

transcript_3_no_greet = """agent(Olivia Martinez) How can I help you today? customer(Jason Brown) Hello, I've got an 
issue with a transaction I didn't authorize. agent(Olivia Martinez) To help you with this, may I have your email and 
the last 4 digits of your SSN? customer(Jason Brown) Sure, my email is jason.brown@mailer.com and the last 4 digits 
are 9988. agent(Olivia Martinez) Thanks, Jason. Can you specify the transaction details, please? customer(Jason 
Brown) It's a withdrawal of $300 on the 20th of July. agent(Olivia Martinez) Okay, I see the transaction. I'll submit 
a dispute form for you right now. You'll receive an email with the confirmation soon. customer(Jason Brown) That's 
great, thank you so much. agent(Olivia Martinez) you are welcome!
"""

docs = [
    Document(
        page_content=transcript_1_positive,
        metadata={"call_id": "359872",
                  "call_date_time": "15:03:2024 14:20:35",
                  "call_duration": "00:08:21",
                  "agent": "Emily Rogers"},
    ),
    Document(
        page_content=transcript_2_no_sec_check,
        metadata={"call_id": "987643",
                  "call_date_time": "17:05:2023 10:45:15",
                  "call_duration": "00:06:54",
                  "agent": "Sarah Bennett"},
    ),
    Document(
        page_content=transcript_3_no_greet,
        metadata={"call_id": "456123",
                  "call_date_time": "22:07:2023 09:30:12",
                  "call_duration": "00:10:30",
                  "agent": "Olivia Martinez"},
    )
]

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

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
    structured_query_translator=ChromaTranslator(),
)

query = """list all the calls where the agent greeted the customer properly?"""

response = retriever.invoke(query)

print(response)
