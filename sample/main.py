import pandas as pd

from dotenv import load_dotenv

load_dotenv()

details = (pd.read_csv("docs/Hotel_details.csv")
           .drop_duplicates(subset="hotelid")
           .set_index("hotelid"))

attributes = pd.read_csv("docs/Hotel_Room_attributes.csv", index_col="id")
price = pd.read_csv("docs/hotels_RoomPrice.csv", index_col="id")

latest_price = price.drop_duplicates(subset="refid", keep="last")[
    [
        "hotelcode",
        "roomtype",
        "onsiterate",
        "roomamenities",
        "maxoccupancy",
        "mealinclusiontype",
    ]
]
latest_price["ratedescription"] = attributes.loc[latest_price.index]["ratedescription"]
latest_price = latest_price.join(
    details[["hotelname", "city", "country", "starrating"]], on="hotelcode"
)
latest_price = latest_price.rename({"ratedescription": "roomdescription"}, axis=1)
latest_price["mealsincluded"] = ~latest_price["mealinclusiontype"].isnull()
latest_price.pop("hotelcode")
latest_price.pop("mealinclusiontype")
latest_price = latest_price.reset_index(drop=True)

#  write header descriptions using GPT-4

from langchain_community.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

attribute_info = model.predict(
    "Below is a table with information about hotel rooms. "
    "Return a JSON list with an entry for each column. Each entry should have "
    '{"name": "column name", "description": "column description", "type": "column data type"}'
    f"\n\n{latest_price.head()}\n\nJSON:\n"
)

import json

attribute_info = json.loads(attribute_info)
print(attribute_info)

from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)

doc_contents = "Detailed description of a hotel room"
prompt = get_query_constructor_prompt(doc_contents, attribute_info)
print(prompt.format(query="{query}"))

chain = load_query_constructor_runnable(
    ChatOpenAI(model="gpt-4-1106-preview", temperature=0), doc_contents, attribute_info
)

response = chain.invoke({"query": "I want a hotel in Southern Europe and my budget is 200 bucks."})

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

from langchain.schema import Document

docs = []

for _, room in latest_price.fillna("").iterrows():
    doc = Document(
        page_content=json.dumps(room.to_dict(), indent=2),
        metadata=room.to_dict()
    )
    docs.append(doc)

vecstore = Chroma.from_documents(docs, embeddings)

from langchain.retrievers import SelfQueryRetriever

retriever = SelfQueryRetriever(
    query_constructor=chain, vectorstore=vecstore, verbose=True
)

results = retriever.get_relevant_documents(
    "I want to stay somewhere highly rated along the coast. I want a room with a patio and a fireplace."
)

for res in results:
    print(res.page_content)
    print("\n" + "-" * 20 + "\n")

