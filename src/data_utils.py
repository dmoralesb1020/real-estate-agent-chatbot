import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List

def load_listings(path: str) -> str:
    """Loads property listings from a text file.

    Args:
        path (str): The path to the text file containing the listings.

    Returns:
        str: The content of the listings file as a single string.

    Raises:
        FileNotFoundError: If the file specified by 'path' is not found.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Listings file not found at path: {path}")
    
def create_vector_store(listings_text: List[str], embedding_model: OpenAIEmbeddings) -> Chroma:
    """Creates a vector store from property listings using ChromaDB.

    This function splits the listings into smaller chunks using a RecursiveCharacterTextSplitter,
    embeds them using the provided embedding model, and stores them in a ChromaDB vector store.

    Args:
        listings_text: A list of property listing strings.
        embedding_model: The OpenAIEmbeddings model to use for generating document embeddings.

    Returns:
        A ChromaDB vector store containing the embedded listings.
    """
    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=20
    )

    # Split the listings into chunks
    chunks = []
    for listing in listings_text:
        chunks.extend(splitter.split_text(listing))

    # Create the ChromaDB vector store
    return Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name="real_estate_listings"
    )
