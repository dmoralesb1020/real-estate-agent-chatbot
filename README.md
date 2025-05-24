# Real Estate Agent Chatbot

This project implements a generative AI chatbot that acts as a friendly and knowledgeable real estate agent. The chatbot utilizes OpenAI's language models and vector search capabilities to understand user queries about real estate preferences and retrieve relevant property listings from a pre-loaded file.

The process involves loading property listings from a text file, storing these listings in a vector database for efficient similarity search, and then building a conversational retrieval chain to interact with the user. The chatbot can understand natural language questions about property requirements and provide answers based on the loaded listings, simulating a real estate agent interaction.

This project was started as the final project for Udacity's Generative AI nanodegree program. The initial version of this project included the capability to generate property listings using a Large Language Model (LLM) and save them to a file. That version is included in the notebook Real_estate_agent.ipynb.


## 🚀 Features

*   **AI-Powered Real Estate Agent:** Interacts with users in a conversational manner, simulating a real estate agent.
*   **Vector Search:** Utilizes vector embeddings and a vector database (Chroma) to efficiently search for relevant listings based on user queries.
*   **Conversational Retrieval:** Employs a Conversational Retrieval Chain (from Langchain) to maintain conversation history and provide contextually relevant responses.


## 🧱 Project Structure

```
├── data/
│ ├── chatbot_prompt.txt
│ └── listings.txt
├── notebooks/
│ └── Real_estate_agent.ipynb
├── src/
│ ├── __init__.py
│ ├── chatbot_app.py
│ ├── data_utils.py
│ ├── llm_utils.py
│ └── prompt_templates.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 🛠️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/dmoralesb1020/real-estate-agent-chatbot.git
cd real-estate-agent-chatbot
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your OpenAI API key**

This project requires your OpenAI API key. You'll need to create a `.env` file in the root directory of the project and add your API key there.

Create a file named `.env` in the real-estate-agent-chatbot directory and add the following, replacing `your_openai_api_key_here` with your actual OpenAI API Key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## ✅ Running the Application

```python src/chatbot_app.py```

The chatbot will load the listings and start a conversation in your terminal.

## 📌 Notes

The chatbot system prompt is in ```data/chatbot_prompt.txt```.

Listings are stored in ```data/listings.txt``` and embedded using OpenAI's embedding model.

## 🧠 Technologies
* Python
* OpenAI GPT & Embeddings
* LangChain
* ChromaDB
* Recursive Text Splitting

## 📫 Contact
Feel free to reach out via moralesb.diego@gmail.com.