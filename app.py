import os
import tempfile
import streamlit as st

from pypdf import PdfReader
OpenAIEmbeddings = None
try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None
Chroma = None
try:
    from langchain.vectorstores import Chroma
except Exception:
    Chroma = None
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        chromadb = None
ChatOpenAI = None
try:
    from langchain.chat_models import ChatOpenAI
except Exception:
    try:
        from langchain.llms import OpenAI as OpenAILLM

        def ChatOpenAI(*args, **kwargs):
            # Adapt call to langchain.llms.OpenAI which expects model_name or model
            model_name = kwargs.pop("model_name", None)
            if model_name is None and len(args) >= 1:
                model_name = args[0]
            temperature = kwargs.pop("temperature", 0.0)
            return OpenAILLM(model_name=model_name, temperature=temperature, **kwargs)
    except Exception:
        ChatOpenAI = None
from langchain.chains import RetrievalQA
from types import SimpleNamespace


st.set_page_config(page_title="PDF Q&A (LangChain + Streamlit)", layout="wide")


def _chunk_text(text, chunk_size=1000, chunk_overlap=200):
    if chunk_size <= 0:
        yield text
        return
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield text[start:end]
        if end == text_len:
            break
        start = max(end - chunk_overlap, end)


def process_pdf(file_bytes, chunk_size=1000, chunk_overlap=200, persist_directory=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    chunks = []
    metadatas = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        for chunk in _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append(chunk)
            metadatas.append({"page": i + 1})

    if OpenAIEmbeddings is None:
        raise ImportError(
            "OpenAIEmbeddings not found in installed langchain.\n"
            "Please pin a compatible langchain version or install a LangChain release that provides OpenAIEmbeddings.\n"
            "Example: add 'langchain>=0.0.300' to requirements.txt or in your environment run: pip install 'langchain>=0.0.300'\n"
            "Alternatively, modify the app to use a different embeddings implementation."
        )

    embeddings = OpenAIEmbeddings()
    # If LangChain's Chroma is available use it directly
    if Chroma is not None:
        vectordb = Chroma.from_texts(chunks, embeddings, metadatas=metadatas, persist_directory=persist_directory)
        return vectordb

    # Fallback: use chromadb directly and wrap a simple retriever interface
    if chromadb is None:
        raise ImportError(
            "No vectorstore available. Install langchain with Chroma support or install chromadb."
        )

    client = chromadb.Client(Settings(chroma_db_impl="inmemory"))
    # create or get a collection
    try:
        collection = client.get_collection(name="pdfreader")
    except Exception:
        collection = client.create_collection(name="pdfreader")

    # compute embeddings for all chunks
    if not hasattr(embeddings, "embed_documents"):
        raise ImportError("Embeddings implementation does not support embed_documents().")
    vectors = embeddings.embed_documents(chunks)

    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=vectors, metadatas=metadatas or [], documents=chunks)

    # simple wrapper to provide as_retriever()
    class SimpleRetriever:
        def __init__(self, collection, embeddings):
            self.collection = collection
            self.embeddings = embeddings

        def as_retriever(self, search_kwargs=None):
            k = 4
            if search_kwargs and "k" in search_kwargs:
                k = search_kwargs["k"]
            return SimpleRetriever.Retriever(self.collection, self.embeddings, k)

        class Retriever:
            def __init__(self, collection, embeddings, k):
                self.collection = collection
                self.embeddings = embeddings
                self.k = k

            def get_relevant_documents(self, query):
                # compute query embedding
                if hasattr(self.embeddings, "embed_query"):
                    qvec = self.embeddings.embed_query(query)
                else:
                    qvec = self.embeddings.embed_documents([query])[0]
                res = self.collection.query(query_embeddings=[qvec], n_results=self.k, include=["documents", "metadatas"])
                docs = []
                # res fields are lists per query
                docs_list = res.get("documents", [[]])[0]
                metas_list = res.get("metadatas", [[]])[0]
                for d, m in zip(docs_list, metas_list):
                    docs.append(SimpleNamespace(page_content=d, metadata=m or {}))
                return docs

    return SimpleRetriever(collection, embeddings)


def get_qa_chain(vectordb, model_name="gpt-3.5-turbo", temperature=0.0):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def main():
    st.title("PDF Q&A — LangChain + Streamlit")

    st.sidebar.header("Settings")
    openai_key = st.sidebar.text_input("OpenAI API key", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)

    uploaded = st.file_uploader("Upload a PDF file", type=["pdf"]) ;

    if uploaded is not None:
        if "vectordb" not in st.session_state or st.session_state.get("uploaded_name") != uploaded.name:
            with st.spinner("Processing PDF and building vector store — this may take a minute..."):
                file_bytes = uploaded.read()
                vectordb = process_pdf(file_bytes, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.session_state["vectordb"] = vectordb
                st.session_state["uploaded_name"] = uploaded.name
        else:
            vectordb = st.session_state["vectordb"]

        st.success(f"Loaded: {uploaded.name}")

        qa_chain = get_qa_chain(vectordb, model_name=model_name, temperature=temperature)

        st.subheader("Ask questions about the PDF")
        question = st.text_input("Your question")
        if st.button("Ask") and question:
            with st.spinner("Generating answer..."):
                result = qa_chain.run(question)
                st.markdown("**Answer**")
                st.write(result)

                # If chain returned source_documents, try to show them
                if isinstance(result, dict) and "source_documents" in result:
                    st.markdown("**Sources**")
                    for d in result["source_documents"]:
                        st.write(d.page_content[:1000])


if __name__ == "__main__":
    main()
