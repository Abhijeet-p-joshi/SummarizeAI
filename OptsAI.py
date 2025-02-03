import streamlit as st
import tempfile
import os
import fitz
import requests
from io import BytesIO
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sentence_transformers import CrossEncoder, SentenceTransformer

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import ollama
import requests
from bs4 import BeautifulSoup

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""
    
def process_document(uploaded_file) -> list[Document]:
    # Ensure uploaded_file is a Streamlit UploadedFile object
    pdf_bytes = uploaded_file.getvalue()  # Use getvalue() to get bytes
    
    # Use PyMuPDF without temp files
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    docs = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        docs.append(Document(
            page_content=text,
            metadata={"source": uploaded_file.name, "page": page_num+1}
        ))
    
    # Same splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return text_splitter.split_documents(docs)

def process_link(link, username=None, password=None):
    try:
        auth = (username, password) if username and password else None
        response = requests.get(link, auth=auth, timeout=10)

        if response.status_code in [401, 403]:
            st.warning("Authentication required. Please enter valid credentials")
            st.session_state.requires_auth = True
            return None

        response.raise_for_status()

        link_parsedData = BeautifulSoup(response.text, "html.parser")

        text_content = ' '.join([p.get_text() for p in link_parsedData.find_all('p')])

        if not text_content:
            st.warning("No readable text found on the provided link.")
            return None

        st.success("Link content processed successfully!")
        return text_content.strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the link: {e}")
        return None    

def get_vector_collection() -> chromadb.Collection: # Contains embedding function
    ollama_embdfun = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    
    chroma_client = chromadb.PersistentClient(path="./demo-optsgpt-chroma")
    return chroma_client.get_or_create_collection(
        name="opts_gpt",
        embedding_function=ollama_embdfun,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
        
    if not documents or not metadatas or not ids:
        st.error("No valid documents found for vector storage.")
        return

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")

def prompt_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

if __name__ == "__main__":
    
    # Session state initialization
    session_defaults = {
        "authenticated": False,
        "requires_auth": False,
        "auth_done": False,
        "username": "",
        "password": "",
        "link_to_process": ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    with st.sidebar: 
        option = st.radio("Choose input type:", ("Document", "Link", "Source Code"))
        
        if option == "Document":  
            uploaded_file = st.file_uploader("**Upload PDF file**", type=["pdf"], accept_multiple_files=False)
        elif option == "Link":  
            link = st.text_input("**Paste the link**")
        elif option == "Source Code": 
            code = st.file_uploader("**Upload source**", type=[".cpp", ".m"], accept_multiple_files=False)

        process = st.button("Process")
    
        if process:
            if option == "Document" and uploaded_file:
                normalize_uploaded_file_name = uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )
                all_splits = process_document(uploaded_file)
                add_to_vector_collection(all_splits, normalize_uploaded_file_name)
                st.success("✅ Document uploaded and processed successfully!")
                
            elif option == "Link" and link:
                st.session_state.link_to_process = link
                
                if st.session_state.requires_auth and not st.session_state.auth_done:
                    # Display authentication form only when authentication is required
                    with st.form("auth_form"):
                        st.session_state.username = st.text_input("Username", key="auth_username")
                        st.session_state.password = st.text_input("Password", type="password", key="auth_password")
                        auth_submit = st.form_submit_button("Authenticate & Process")

                    if auth_submit:
                        st.session_state.auth_done = True
                        processed_text = process_link(
                            st.session_state.link_to_process,
                            st.session_state.username,
                            st.session_state.password
                        )
                        if processed_text:
                            # Create and process document
                            link_doc = [Document(
                                page_content=processed_text,
                                metadata={"source": link, "page": 1}
                            )]
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=500,
                                chunk_overlap=100,
                                separators=["\n\n", "\n", ".", "?", "!", " ", ""]
                            )
                            link_splits = text_splitter.split_documents(link_doc)
                            normalized_name = link.translate(str.maketrans({"-": "_", ".": "_", " ": "_", "/": "_", ":": "_"}))
                            add_to_vector_collection(link_splits, normalized_name)
                            st.success("✅ Link content processed and stored!")
                else:
                    processed_text = process_link(
                        st.session_state.link_to_process,
                        st.session_state.username if st.session_state.auth_done else None,
                        st.session_state.password if st.session_state.auth_done else None
                    )
                    if processed_text:
                        # Create document from link content
                        link_doc = [Document(
                            page_content=processed_text,
                            metadata={"source": link, "page": 1}
                        )]
                        # Process with same splitter as PDF
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
                        )
                        link_splits = text_splitter.split_documents(link_doc)
                        # Normalize link name for storage
                        normalized_name = link.translate(str.maketrans({"-": "_", ".": "_", " ": "_", "/": "_", ":": "_"}))
                        add_to_vector_collection(link_splits, normalized_name)
                        st.success("✅ Link content processed and stored!")
            
    # Question and Answer Area
    st.header("OPTS GPT")
    prompt = st.text_area("**Ask a question about OPTS:**")
    ask = st.button( "**Ask**")
    
    if ask and prompt:
        results = prompt_collection(prompt)
        context = results.get("documents")[0]
        relevant_context, relevant_context_ids = re_rank_cross_encoders(context)
        # response = call_llm(context=context, prompt=prompt)
        response = call_llm(context=relevant_context, prompt=prompt)
        st.write_stream(response)
        
        with st.expander("See retrieved documents"):
            st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_context_ids)
            st.write(relevant_context)