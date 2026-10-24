
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("Step 1: Loading policy...", flush=True)

file_path = Path(__file__).with_name("leave_policy.txt")
documents = TextLoader(
    str(file_path),
    encoding="utf-8"
).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}", flush=True)

print("Step 2: Creating embeddings...", flush=True)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Step 3: Creating vector database...", flush=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="employee_policies"
)

print("Step 4: Searching by meaning...", flush=True)

question = "How much sick leave do employees get?"

results = vector_store.similarity_search(
    question,
    k=1
)

print("\nSearch results:", flush=True)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))

print("\nSemantic search completed!", flush=True)
