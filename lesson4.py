
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "leave_policy.txt"
DB_DIR = BASE_DIR / "chroma_db"

print("Step 1: Loading policy...", flush=True)

documents = TextLoader(
    str(POLICY_FILE),
    encoding="utf-8"
).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.", flush=True)

print("Step 2: Initializing embedding model...", flush=True)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Step 3: Building/loading Chroma database...", flush=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="employee_policies",
    persist_directory=str(DB_DIR)
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

print("Step 4: Creating RAG chain...", flush=True)

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an employee helpdesk assistant.

Answer the question using only the policy context below.
If the answer is not present in the context, say:
"I couldn't find that in the provided policy."

Policy context:
{context}

Employee question:
{question}

Answer:
""")

def format_documents(documents):
    return "\n\n".join(
        doc.page_content for doc in documents
    )

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": lambda question: question
    }
    | prompt
    | model
    | StrOutputParser()
)

question = "Who approves leave requests?"

print("\nQuestion:", question, flush=True)
print("\nSearching policy and generating answer...", flush=True)

answer = rag_chain.invoke(question)

print("\nAssistant:", answer, flush=True)
