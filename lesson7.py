from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "leave_policy.txt"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "employee_policies_persistent"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)

if vector_store._collection.count() == 0:
    print("Collection is empty. Indexing policy documents...")

    documents = TextLoader(
        str(POLICY_FILE),
        encoding="utf-8",
    ).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )

    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)

    print(f"Indexed {len(chunks)} chunks.")
else:
    print("Existing documents found. Reusing the collection.")

print("Stored chunks:", vector_store._collection.count())

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

while True:
    question = input(
        "\nAsk about the leave policy (or type exit): "
    ).strip()

    if question.lower() in {"exit", "quit"}:
        break

    if not question:
        continue

    results = retriever.invoke(question)

    print("\nRetrieved policy context:")
    for i, doc in enumerate(results, start=1):
        print(f"\n[{i}] {doc.page_content}")