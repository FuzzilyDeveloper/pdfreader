# PDF Q&A — LangChain + Streamlit

A small Streamlit app that loads a PDF, builds an embeddings-based vector store with LangChain, and answers questions using an OpenAI LLM.

Requirements

- Python 3.10+
- See `requirements.txt` for Python packages.

Installation

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Set your OpenAI API key (or paste it into the sidebar when running the app):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Run

```bash
streamlit run "app.py"
```

Usage

- Upload a PDF in the app UI.
- Wait for the vectorstore to build.
- Ask questions in the text input and click "Ask".

Notes & Next steps

- This sample uses `Chroma` for an in-memory vector store.
- For production or larger PDFs, consider persisting the vector store (`persist_directory`) or using an external vector DB.
- To deploy: consider Streamlit Cloud, Docker containers, or a simple VM.
