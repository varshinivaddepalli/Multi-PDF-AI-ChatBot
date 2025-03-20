# 📚 Multi-PDF AI ChatBot

## Overview
The goal of this project is to create a user-centric and intelligent system that enhances information retrieval from PDF documents through natural language queries. The project focuses on streamlining the user experience by developing an intuitive interface, allowing users to interact with PDF content using language they are comfortable with. To achieve this, we leverage the **Retrieval Augmented Generation (RAG)** methodology introduced by Meta AI researchers.

---

## 🔹 Retrieval Augmented Generation (RAG)

### **Introduction**
RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an **information retrieval component** with a **text generator model** to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for **fine-tuning and modification** of internal knowledge without extensive retraining.

### **Workflow**
1️⃣ **Input:** RAG takes multiple PDFs as input.
2️⃣ **VectorStore:** PDFs are converted to **FAISS vector store** using the `all-MiniLM-L6-v2` embeddings model from Hugging Face.
3️⃣ **Memory:** A **conversation buffer memory** maintains previous conversations, which are fed into the LLM along with the user query.
4️⃣ **Text Generation with Mistral-7B:** The embedded input is processed by **Mistral-7B-v0.1** via the Hugging Face API to generate responses.
5️⃣ **User Interface:** Streamlit provides an interactive and user-friendly UI for the application.

---

## 🚀 Features

✅ **Extracts text from PDFs** (text-based & scanned using OCR)
✅ **Vectorizes text** using FAISS and Hugging Face embeddings
✅ **Retrieves contextually relevant answers** using a conversational AI model
✅ **Handles multi-page PDFs** with efficient text splitting
✅ **Interactive UI with Streamlit** for easy PDF uploads & querying
✅ **Supports multi-PDF processing for comprehensive analysis**

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Streamlit** (for UI)
- **LangChain** (Conversational Retrieval Chain)
- **FAISS** (Vector search)
- **Hugging Face Transformers**
- **Mistral-7B** (LLM for answering queries)
- **pdfplumber & pytesseract** (Text & OCR extraction)

---

## 📦 Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/Multi-PDF-AI-ChatBot.git
cd Multi-PDF-AI-ChatBot
```

2️⃣ Install dependencies

3️⃣ Set up Hugging Face API key (Optional for hosted inference):
```bash
export HUGGINGFACEHUB_API_TOKEN="your_api_key_here"
```

4️⃣ Run the chatbot:
```bash
streamlit run app.py
```

---

## 📂 Usage

1. Upload one or more PDFs via the Streamlit sidebar.
2. Click **Process** to extract text & build vector embeddings.
3. Ask questions in the chat input field.
4. Get real-time, AI-generated answers based on your document!

---

## 🎯 Benefits

✅ **Adaptability:** RAG adapts to evolving knowledge domains, making it ideal for dynamic information retrieval.
✅ **Efficiency:** By combining retrieval and generation, RAG provides access to the latest information without extensive retraining.
✅ **Reliability:** The methodology ensures **reliable outputs** by leveraging both retrieval-based and generative approaches.
✅ **Seamless Navigation:** The system streamlines information retrieval, reducing complexity and enhancing the overall user experience.

---

## 🛠 Future Enhancements

- ✅ Add local/offline LLMs (e.g., `llama.cpp` or `GPT4All`) to remove API dependency
- ✅ Improve UI with a real-time chat history sidebar
- ✅ Optimize large PDF processing with streaming methods
- ✅ Enhance error handling for corrupt/empty PDFs

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## ⭐ Acknowledgments

Special thanks to **Hugging Face**, **LangChain**, and **FAISS** for their open-source contributions that power this chatbot!

---

Enjoy using **Multi-PDF AI ChatBot**! 🚀
