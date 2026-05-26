from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "notes.txt", "page": 1}
    ),

    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "notes.txt", "page": 2}
    ),

    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "notes.txt", "page": 3}
    )
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory= "chrome-db" #creating a local storage to save the data, hence creating directory
)

result = vectorstore.similarity_search("what is used for data analysis",k=2)#vectorstore is only responsible for retrieving your information
#llm is responsible for answering your question
#k represents how many results/documents embeddings do you want

for r in result:
    print (r)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)