# Multi-PDF-AI-ChatBot


## Overview

The goal of this project is to create a user-centric and intelligent system that enhances information retrieval from PDF documents through natural language queries. The project focuses on streamlining the user experience by developing an intuitive interface, allowing users to interact with PDF content using language they are comfortable with. To achieve this, we leverage the Retrieval Augmented Generation (RAG) methodology introduced by Meta AI researchers.



## Retrieval Augmented Generation (RAG)

### Introduction

RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an information retrieval component with a text generator model to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for fine-tuning and modification of internal knowledge without extensive retraining.

### Workflow

1. **Input**: RAG takes multiple pdf as input.
2. **VectoreStore**: The pdf's are then converted to vectorstore using FAISS and all-MiniLM-L6-v2 Embeddings model from Hugging Face.
3. **Memory**: Conversation buffer memory is used to maintain a track of previous conversation which are fed to the llm model along with the user query.
4. **Text Generation with Mistral-7B-v0.1**: The embedded input is fed to the Mistral-7B-v0.1 model from the Huggingface API, which produces the final output.
5. **User Interface**: Streamlit is used to create the interface for the application.

### Benefits

- **Adaptability**: RAG adapts to situations where facts may evolve over time, making it suitable for dynamic knowledge domains.
- **Efficiency**: By combining retrieval and generation, RAG provides access to the latest information without the need for extensive model retraining.
- **Reliability**: The methodology ensures reliable outputs by leveraging both retrieval-based and generative approaches.

## Project Features

1. **User-friendly Interface**: An intuitive interface designed to accommodate natural language queries, simplifying the interaction with PDF documents.

2. **Seamless Navigation**: The system streamlines information retrieval, reducing complexity and enhancing the overall user experience.
