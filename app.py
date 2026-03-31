from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

@st.cache_resource #Loads the chain once and stores it insted of calling it again and again
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
st.caption("Ask anything about your uploaded document")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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