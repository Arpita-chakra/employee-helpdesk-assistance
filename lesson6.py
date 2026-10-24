
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "leave_policy.txt"
DB_DIR = BASE_DIR / "chroma_db"

# 1. Load and split the policy.
documents = TextLoader(
    str(POLICY_FILE),
    encoding="utf-8"
).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
chunks = splitter.split_documents(documents)

# 2. Create embeddings and vector store.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="employee_policies_chat_v2",
    persist_directory=str(DB_DIR)
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 3. Local LLM.
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Keep history as LangChain messages.
chat_history = []

# 4. Rewrite follow-up questions using history.
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the latest question as a standalone search query.
Use the chat history to resolve references such as 'it', 'those',
and 'how many can I carry forward'.
If the question is already standalone, preserve its meaning.
Do not answer the question. Return only the rewritten query."""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{question}")
])

rewrite_chain = rewrite_prompt | model | StrOutputParser()

# 5. Generate answers using retrieved policy text.
answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an employee policy assistant.
Answer using only the supplied policy context.
Use chat history to understand the conversation, but do not
treat previous assistant answers as authoritative policy.
If the policy context does not support an answer, say:
'I couldn't find that in the provided policy.'
Never invent company rules."""
    ),
    ("placeholder", "{chat_history}"),
    (
        "human",
        """Policy context:
{context}

Current question:
{question}

Answer:"""
    )
])

answer_chain = answer_prompt | model | StrOutputParser()


def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question):
    # A. Rewrite the query using conversation history.
    standalone_question = rewrite_chain.invoke({
        "chat_history": chat_history,
        "question": question
    }).strip()

    print(f"\nSearch query: {standalone_question}")

    # B. Retrieve relevant policy chunks.
    source_docs = retriever.invoke(standalone_question)

    # C. Generate an answer grounded in retrieved context.
    answer = answer_chain.invoke({
        "chat_history": chat_history,
        "context": format_documents(source_docs),
        "question": question
    })

    # D. Update history after both calls complete.
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer)
    ])

    print(f"\nAssistant: {answer}")
    print("\nSources:")
    for i, doc in enumerate(source_docs, start=1):
        print(f"[{i}] {doc.page_content}")


print("Employee Knowledge Assistant — Conversational RAG")
print("Type 'exit' to quit or 'reset' to clear conversation history.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if question.lower() == "reset":
        chat_history.clear()
        print("Conversation history cleared.")
        continue

    if not question:
        continue

    try:
        answer_question(question)
    except Exception as error:
        print(f"An error occurred: {error}")
