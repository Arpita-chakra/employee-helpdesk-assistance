
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Step 1: Imports successful!", flush=True)

# Use a local model instead of the OpenAI API
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an employee helpdesk assistant. "
        "Answer using only the supplied policy. "
        "If information is missing, say you cannot find it."
    ),
    (
        "human",
        "Policy: {context}\nQuestion: {question}"
    )
])

chain = prompt | model | StrOutputParser()

print("Step 2: Chain created!", flush=True)

response = chain.invoke({
    "context": (
        "Employees receive 18 annual leaves. "
        "A maximum of 5 unused leaves can be carried forward."
    ),
    "question": "How many leaves can I carry forward?"
})

print("\nAssistant:", response, flush=True)
