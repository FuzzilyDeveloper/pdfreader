import os
import tempfile
import streamlit as st
import langchain


from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="PDF Q&A (LangChain + Streamlit)", layout="wide")


# Custom simple text splitter as RecursiveCharacterTextSplitter is unavailable
def simple_text_splitter(docs, chunk_size=1000, chunk_overlap=200):
    split_docs = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            split_docs.append(type(doc)(page_content=chunk))
            start += chunk_size - chunk_overlap
    return split_docs


def process_pdf(file_bytes, chunk_size=1000, chunk_overlap=200, persist_directory=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Use custom splitter
    docs_split = simple_text_splitter(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=persist_directory)
    return vectordb


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
