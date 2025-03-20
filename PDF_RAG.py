# Import dependencies
import os
import torch
import streamlit as st
import pdfplumber
import fitz  # PyMuPDF for better PDF processing
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from PIL import Image

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_key:
    raise ValueError("Hugging Face API key is missing or not loaded properly!")

print("API Key Loaded Successfully!")  # Debugging step

# Custom template for rephrasing questions
custom_template = """Given the following conversation and a follow-up question, 
rephrase the follow-up question to be a standalone question in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Function to extract text from PDFs (supports scanned and text-based PDFs)
def extract_text_from_pdfs(docs):
    text = ""
    for pdf in docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
                else:
                    # OCR for scanned PDFs
                    images = convert_from_path(pdf)
                    for image in images:
                        text += pytesseract.image_to_string(image) + "\n"

                # Extract tables from PDFs safely
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:  # Ensure row is not None
                            clean_row = [cell if cell else "" for cell in row]  # Replace None with empty string
                            text += " | ".join(clean_row) + "\n"
    
    return text.strip()

# Function to split text into chunks for better retrieval
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create FAISS vector store using sentence embeddings
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",  # More accurate for technical/legal docs
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-v0.1",
        model_kwargs={"temperature": 0.2, "max_length": 1024},
        huggingfacehub_api_token=hf_api_key
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )

# Function to handle user queries
def handle_question(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a document before asking questions.")
        return
    
    response = st.session_state.conversation({'question': question})
    helpful_answer = response.get('answer', "").strip()

    # Clean unnecessary labels
    for prefix in ["Helpful Answer:", "Answer:"]:
        if helpful_answer.startswith(prefix):
            helpful_answer = helpful_answer[len(prefix):].strip()

    st.write(helpful_answer)

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF POOKIE")

    question = st.text_input("I read PDFs so you don’t have to… because I care. 😉")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Upload your PDFs")
        docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True, type=['pdf'])

        if st.button("Process"):
            if not docs:
                st.warning("Please upload at least one document before processing.")
            else:
                with st.spinner("Extracting and processing text..."):
                    raw_text = extract_text_from_pdfs(docs)
                    
                    if not raw_text:
                        st.error("No text could be extracted from the PDFs. Try different files.")
                        return
                    
                    text_chunks = get_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success("Processing complete! You can now ask questions.")

if __name__ == '__main__':
    main()
