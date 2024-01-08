from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

loader = DirectoryLoader('/Users/chamindawijayasundara/Documents/research/data_analysis/self_query_retreaver'
                         '/call_filter/plain/json',
                         glob='**/*.json',
                         show_progress=True,
                         loader_cls=TextLoader,
                         use_multithreading=True)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0)

texts = text_splitter.split_documents(docs)

persist_directory = './vectorstore'

vector_store = Chroma.from_documents(documents=texts,
                                     embedding=embeddings,
                                     persist_directory=persist_directory)



