from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from ingest import build_index
import streamlit as st
import tempfile
import os

load_dotenv()

@st.cache_resource  # Loads the chain once and stores it instead of calling it again and again
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# UI
st.title("RAG Document Chatbot")

# Sidebar — PDF upload
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        current_name = st.session_state.get("loaded_pdf")
        if current_name != uploaded_file.name:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    build_index(tmp_path)
                    st.session_state.loaded_pdf = uploaded_file.name
                    st.session_state.messages = []
                    load_chain.clear()
                    st.success(f"Indexed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.get("loaded_pdf"):
        st.caption(f"Active PDF: **{st.session_state.loaded_pdf}**")
    elif os.path.exists("faiss_index"):
        st.caption("Using existing index.")
    else:
        st.warning("No PDF loaded. Upload a PDF to get started.")

st.caption("Ask anything about your uploaded document")

index_ready = os.path.exists("faiss_index")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if index_ready:
    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = load_chain()
                result = chain.invoke({"query": query})
                answer = result["result"]
                sources = result["source_documents"]

            st.write(answer)

            with st.expander("Sources used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** (page {doc.metadata.get('page', '?')+1})")
                    st.caption(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF in the sidebar to start chatting.")
