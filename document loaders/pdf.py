from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/notes (1).pdf")

docs=data.load()

print(docs[1])