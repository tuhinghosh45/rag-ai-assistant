from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

#embedding_model = MistralAIEmbeddings(model="mistral-embed")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  
)


vectorstore= Chroma(
    embedding_function= embedding_model,
    persist_directory= "chroma-db"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10, 
        "lambda_mult": 0.5 #0 means more diverse result, 1 means very less diverse result
    }
    )

llm = init_chat_model("mistral-small-latest")


#prompt template

prompt = ChatPromptTemplate.from_messages(
    [("system", """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
      ),
     (
            "human",
            """Context:
        {context}

        Question:
        {question}
"""
        )]
)

print("Rag system created")

print("Press 0 to exit")

while True:
    query=input("You: ")
    if query == '0':
        break

    docs= retriever.invoke(query)

    context= "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt= prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response= llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")







