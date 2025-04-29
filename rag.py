# Importing Langchain and chroma And plotly
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

import numpy as np

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import glob
from dotenv import load_dotenv
import gradio as gr
import shutil
import gradio as gr

load_dotenv()


MODEL = "gpt-4o-mini"
db_name = "vector-db"
conversation_chain = None


def build_vector_store(folders):

    global conversation_chain
    global db_name
    global MODEL
    # print(f"Building vector store from {folder}\n=====================================")

    # Read in Documents using Langchain's loader
    # Take everthing in all the sub-folders of our knowledgebase
    # Loaders for various file types
    text_loader_kwargs = {"encoding": "utf-8"}
    documents = []

    for file in folders:
        doc_type = os.path.basename(file).split(".")[-1]
        print(f"Document type: {doc_type}")
        # Load Markdown files (.md)
        if doc_type == "md":

            for doc in TextLoader(file, encoding="utf-8").load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)

        # Load PDF files (.pdf)
        elif doc_type == "pdf":

            for doc in PyPDFLoader(file).load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        elif doc_type == "docx":
            # Load Word documents (.docx)

            for doc in UnstructuredWordDocumentLoader(file).load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        elif doc_type == "txt":
            # Load plain text files (.txt)

            for doc in TextLoader(file, encoding="utf-8").load():
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)
        else:
            # Splitting text into chunks
            gr.Warning("Sahi file daal lode")
            return

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    # print(f"Document types found  {', '.join(doc_types)}")

    doc_types = set(chunk.metadata["doc_type"] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")

    # Put chunks of data into the vector store that associates a vector embeddings with each chunks

    embeddings = OpenAIEmbeddings()

    # Check if chroma datastore already exists , if so delete the collection to start from scratch

    if os.path.exists(db_name):
        Chroma(
            persist_directory=db_name, embedding_function=embeddings
        ).delete_collection()

    # Create A new Chroma vectorstore

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=db_name
    )
    print(f"Vector Store created with {vectorstore._collection.count()} documents")

    # Let's investigate the vectors

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )

    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # the retriever is an abstraction over the VectorStore that will be used during RAG
    retriever = vectorstore.as_retriever()

    # putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    # set up a new conversation memory for the chat
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
