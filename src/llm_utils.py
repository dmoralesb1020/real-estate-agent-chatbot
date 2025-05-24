import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("🔒 OPENAI_API_KEY not found in .env or environment.")

def get_openai_chat_model(model_name: str = 'gpt-4o-mini', temperature: float = 0.0, max_tokens: int = None):
    """Initializes and returns a ChatOpenAI model."""
    return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

def get_openai_embeddings_model():
    """Initializes and returns an OpenAIEmbeddings model."""
    return OpenAIEmbeddings()

def setup_retriever(vector_store):
    """Creates and returns a retriever from a given vector store."""
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4}
    )

def setup_conversational_chain(retriever, prompt_template: PromptTemplate, model_name: str = 'gpt-4o-mini'):
    """
    Sets up and returns a ConversationalRetrievalChain.

    Args:
        retriever: The Langchain retriever for document retrieval.
        prompt_template (PromptTemplate): The prompt template for the combine_docs_chain.
        model_name (str): The name of the LLM model to use for the conversation.

    Returns:
        ConversationalRetrievalChain: The configured conversational chain.
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key="answer")
    retrieval_llm = get_openai_chat_model(model_name=model_name, temperature=0.0, max_tokens=400)

    chain = ConversationalRetrievalChain.from_llm(
        llm=retrieval_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt_template},
        return_source_documents=True
    )
    return chain
