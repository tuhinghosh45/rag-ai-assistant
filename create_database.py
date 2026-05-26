# load pdf
# split into chunks
# create the embeddings
# store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/DeepLearningMaterial.pdf")#loading the data here first which is notes.txt
docs=data.load()#converting notes.txt to document and creating document object

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

#embedding_model = MistralAIEmbeddings(model="mistral-embed")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  
)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory= "chroma-db" #creating a local storage to save the data, hence creating directory
)