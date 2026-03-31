from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_index(pdf_path: str, index_path: str = "faiss_index"):
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load() # load all the pages of the pdf using Langchain pdf_loader
    print(f"Loaded {len(pages)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50 # Overlap important for context
    )
    chunks = splitter.split_documents(pages)# get chunks off of the pages
    print(f"Created {len(chunks)} chunks")

    print("Building embeddings and FAISS index...") #using FAISS for faster query vector to similar chunk lookup
    transformer = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ) # create embeddings for all the chunks using a sentence transformer not a typical token transformer

    #the sentence transformer is the NN model that takes in a sentence and makes a vector
    vectorstore = FAISS.from_documents(chunks, transformer) # give documents and the transformer to use to FAISS to make a vecotre store of all the sentences in the document
    #remember that the imports are classes and we usually work with objects of those classes using functions in the classes we imported

    print(f"Saving index to {index_path}...") #store the FAISS vectore store to computer
    vectorstore.save_local(index_path)
    print("Done! Index saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py your_document.pdf") #need a pdf argument
    else:
        build_index(sys.argv[1])