
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Step 1: Loading policy document...", flush=True)

# Find the file relative to this Python script.
file_path = Path(__file__).with_name("leave_policy.txt")

# Load the document.
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")

# Inspect the first loaded document.
print("\nOriginal document preview:")
print(documents[0].page_content[:300])

print("\nStep 2: Splitting the document...", flush=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

chunks = splitter.split_documents(documents)

print(f"\nTotal chunks created: {len(chunks)}")

# Inspect every chunk.
for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    print("Characters:", len(chunk.page_content))

print("\nStep 3: Document processing complete!", flush=True)
