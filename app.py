import streamlit as st
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ---------------- LOAD ENV ----------------

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0E1117;
}

h1 {
    color: white;
    font-size: 3rem;
}

.stChatMessage {
    border-radius: 15px;
    padding: 12px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("📚 RAG AI Assistant")

st.markdown(
    "Upload a PDF and chat with your document using Mistral AI + HuggingFace Embeddings."
)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    process = st.button("🚀 Process PDF")

# ---------------- EMBEDDING MODEL ----------------

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---------------- LLM ----------------

llm = init_chat_model(
    "mistral-small-latest"
)

# ---------------- SESSION STATE ----------------

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# ---------------- PROCESS PDF ----------------

if uploaded_file and process:

    with st.spinner("Processing PDF and creating embeddings..."):

        # Save uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")

        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Create unique DB name
        safe_name = uploaded_file.name.replace(" ", "_")

        db_path = f"chroma-db-{safe_name}"

        # Create vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_path
        )

        st.session_state.vectorstore_ready = True
        st.session_state.db_path = db_path

    st.success("✅ PDF processed successfully!")

# ---------------- LOAD VECTORSTORE ----------------

if st.session_state.vectorstore_ready:

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=st.session_state.db_path
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # ---------------- CHAT INPUT ----------------

    user_query = st.chat_input(
        "Ask a question about the document..."
    )

    if user_query:

        # User message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve documents
        docs = retriever.invoke(user_query)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not found in the context,
say:
"I could not find the answer in the document."
"""
                ),

                (
                    "human",
                    """
Context:
{context}

Question:
{question}
"""
                )
            ]
        )

        # Final prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": user_query
            }
        )

        # LLM response
        response = llm.invoke(final_prompt)

        # Assistant response
        with st.chat_message("assistant"):
            st.markdown(response.content)

else:

    st.info("📄 Upload and process a PDF to begin chatting.")