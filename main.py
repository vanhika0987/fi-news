import os
import pickle
import time
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Sidebar input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Main content placeholder
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Processing URLs
if process_url_clicked:
    main_placeholder.text("Data Loading... 🔄")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    if data:
        main_placeholder.text("Text Splitting... 🔄")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        if docs:
            main_placeholder.text("Building Embeddings... 🔄")
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            # Save FAISS index to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            main_placeholder.text("Embeddings Built and Saved! ✅")

        else:
            main_placeholder.error("Text Splitter produced empty documents. Check data or configuration.")

    else:
        main_placeholder.error("Data loading failed. Check URLs or network connection.")

# Querying with loaded FAISS index
query = main_placeholder.text_input("Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display result
            st.header("Answer:")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        main_placeholder.error("FAISS index file not found. Process URLs first.")

