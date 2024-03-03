import os, shutil
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import TextLoader



from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)




def save_to_txt(text):
    try:
        with open("text_input.txt", "w") as file:
            file.write(text)
        st.success("Text saved successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")


st.image('Loreon Tech coloured.png',width=120 )
st.title("Generative AI - Text to Chat")
st.sidebar.title("Text Content")


# text_filee = st.sidebar.text_input(f"Enter your text data here")

# Get text input from the user
text_input = st.sidebar.text_area("Enter your text here:")




process_url_clicked = st.sidebar.button("Process Text Data")

# st.sidebar.button("Process Text Data")

faiss_file_path = "faiss_index"
embeddings = OpenAIEmbeddings()

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:

 # Save the text input to a text file when the user clicks the button
if process_url_clicked:
    save_to_txt(text_input)

    # load data
    loader = TextLoader("text_input.txt")
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    # embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a file
    
    try:
        # # Check if the file already exists, and remove it if it does
        if os.path.exists(faiss_file_path):
            shutil.rmtree(faiss_file_path)

        # Save the FAISS index to the file
        vectorstore_openai.save_local(faiss_file_path)
        main_placeholder.text("Embedding Vector Building Complete...✅✅✅")
    except Exception as e:
        main_placeholder.error(f"An error occurred while saving FAISS index: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(faiss_file_path):
        vectorstore = FAISS.load_local(faiss_file_path, embeddings)
        # with open(file_path, "rb") as f:
        #     vectorstore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)




