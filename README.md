# Employee Helpdesk Assistant

A small Python learning project that explores answering employee leave-policy questions with local language models, LangChain, and Chroma. The policy content is in `leave_policy.txt`; later lessons retrieve relevant policy passages before answering.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- The Ollama models used by the project:

  ```powershell
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python packages used by the lessons:

```powershell
python -m pip install langchain-core langchain-ollama langchain-community langchain-text-splitters langchain-chroma langchain-classic python-dotenv
```

Make sure Ollama is running before launching a script. The model calls use Ollama locally; an OpenAI API key is not required.

## Run

The scripts are standalone lessons that build on each other:

| Script | Description |
| --- | --- |
| `main.py` | Basic prompt-and-model example with a short policy context embedded in the script. |
| `lesson2.py` | Loads and splits `leave_policy.txt` into text chunks. |
| `lesson3.py` | Creates embeddings and performs a semantic search over the policy. |
| `lesson4.py` | Builds a retrieval-augmented generation (RAG) chain and answers a sample question. |
| `lesson5.py` | Interactive policy Q&A with retrieved source passages. |
| `lesson6.py` | Interactive Q&A with follow-up question handling and a resettable conversation history. |
| `lesson7.py` | Stores policy chunks in a persistent local Chroma database and lets you search them. |
| `lesson8.py` | Uses the persistent policy database in an interactive agent with policy search and a leave-balance calculator. |
| `test_key.py` | Checks whether an `OPENAI_API_KEY` is present in `.env`; it is not needed by the Ollama lessons. |

Run any lesson from the project directory, for example:

```powershell
python lesson7.py
```

For the agent in `lesson8.py`, run `lesson7.py` first to populate the persistent Chroma collection. Then start the agent:

```powershell
python lesson8.py
```

Type `exit` or `quit` to leave interactive lessons. The local `chroma_db/` directory stores the vector database and can be regenerated; it is ignored by Git.

## Project files

- `leave_policy.txt` — sample employee leave policy used by the retrieval lessons.
- `lesson*.py` — incremental examples of document loading, retrieval, RAG, conversation history, and tool use.
- `chroma_db/` — generated local vector database created by the persistent lessons.
