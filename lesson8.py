
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# Connect to the same collection created in lesson7.py.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="employee_policies_persistent",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)
print("Database path:", DB_DIR)
print("Collection name:", vector_store._collection.name)
print("Number of stored chunks:", vector_store._collection.count())

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


@tool
def search_leave_policy(question: str) -> str:
    """Search the employee leave policy for rules about annual,
    sick, casual, carry-forward, and leave approval."""
    documents = retriever.invoke(question)

    if not documents:
        return "No relevant policy documents were found."

    return "\n\n".join(doc.page_content for doc in documents)


@tool
def calculate_leave_balance(
    total_days: int,
    days_used: int,
) -> str:
    """Calculate remaining leave days when the user provides
    the total allowance and number of days already used."""
    if total_days < 0 or days_used < 0:
        return "Error: leave days cannot be negative."

    if days_used > total_days:
        return "Error: days used cannot exceed the total allowance."

    remaining = total_days - days_used
    return (
        f"Total allowance: {total_days} days. "
        f"Used: {days_used} days. "
        f"Remaining: {remaining} days."
    )


tools = [
    search_leave_policy,
    calculate_leave_balance,
]

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an employee leave-policy assistant.

Use search_leave_policy for questions about company policy.
Use calculate_leave_balance when the user supplies the total
leave allowance and days used.

Never invent policy rules or personal employee records.
The calculator only performs arithmetic; it does not know
an employee's actual leave balance.

If required information is missing, ask the user for it.
Base policy answers on retrieved policy text.
"""
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

print("Employee Leave Agent")
print("Type 'exit' to quit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    try:
        result = agent_executor.invoke({"input": question})
        print("\nAssistant:", result["output"])
    except Exception as error:
        print(f"\nError: {error}")
