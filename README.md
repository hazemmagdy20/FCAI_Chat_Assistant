# 🎓 FCAI Bylaw Assistant Chatbot

This is a **Flask-based web application** that provides an interactive chat assistant trained on the **academic bylaws** of the **Faculty of Computer and Artificial Intelligence (FCAI)** at **Cairo University**.

It allows students, staff, or anyone interested to ask questions about the academic regulations and receive clear, context-aware answers in **Arabic or English**, depending on the question’s language.

---

## 📌 Features

✅ Retrieve relevant sections from the faculty bylaws  
✅ Generate smart, context-based answers using a powerful LLM  
✅ Supports Arabic and English answers  
✅ Clean, user-friendly chat interface built with HTML & CSS  
✅ Powered by **LangChain**, **FAISS**, and **Hugging Face**

---

## ⚙️ How It Works

1. 📄 **Loads and flattens FCAI bylaw JSON files**
2. 🧩 **Splits text into chunks and builds a FAISS vector store**
3. 🧠 **Retrieves relevant sections based on user queries**
4. 🤖 **Calls a large language model (LLM) to generate answers using the retrieved context**
5. 💬 **Displays answers in a web chat interface**

## How to use

1. Install all required packages using pip install -r requirements.txt
2. Run the Flask server using python app.py




