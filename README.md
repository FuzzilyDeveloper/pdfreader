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

Streamlit Cloud deployment

1. Push your repository to GitHub (already done for this project).
2. Open https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose the `FuzzilyDeveloper/pdfreader` repository and branch, and set the main file to `app.py`.
4. Add your OpenAI API key in the app settings (Secrets):

	- Go to your app on Streamlit Cloud → **Settings** → **Secrets**.
	- Add a key `OPENAI_API_KEY` with your OpenAI key as the value.

	Alternatively, for local testing create a file `.streamlit/secrets.toml` with the following (do not commit it):

	```toml
	OPENAI_API_KEY = "sk-..."
	```

5. Deploy — Streamlit Cloud will install packages from `requirements.txt` and run `app.py`.

Repository URL: https://github.com/FuzzilyDeveloper/pdfreader
