from src.data_utils import load_listings, create_vector_store
from src.llm_utils import get_openai_embeddings_model, setup_retriever, setup_conversational_chain
from src.prompt_templates import load_prompt 
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

def run_chat(chain: ConversationalRetrievalChain) -> None:
    """Runs a real estate chatbot that allows users to query real estate listings.

    The chatbot loads listings from a text file, creates a Chroma vector store,
    creates a ConversationalRetrievalChain, and then allows the user to input
    queries about real estate.

    Args:
        chain: The ConversationalRetrievalChain to use for the chatbot.
    """
    print("=" * 50)
    print("🏡 Real Estate Assistant Chatbot")
    print("Type your preferences to find listings.")
    print("Example: 'I'm looking for a place with 3 bedrooms and a backyard.'")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 50)

    try:
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting the assistant. Goodbye!")
                break

            response = chain.invoke({"question": query})
            print(f"\nAssistant: {response['answer']}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    model_name = 'gpt-4o-mini'
    listings_path = 'data/listings.txt'
    prompt_path = 'data/chatbot_prompt.txt'

    # Load listings
    full_text = load_listings(listings_path)
    listings = [listing.strip() for listing in full_text.split("=== Listing ===") if listing.strip()]
    # Create embedding model and vector store 
    embedding_model = get_openai_embeddings_model()
    vector_store = create_vector_store(listings, embedding_model) 
    # Setup retriever 
    retriever = setup_retriever(vector_store)
    # Load prompt and setup chain
    prompt = load_prompt(prompt_path)
    prompt_template = PromptTemplate.from_template(prompt)
    chain = setup_conversational_chain(
        retriever=retriever,
        prompt_template=prompt_template,
        model_name=model_name # Pass the model name to the chain setup
    )

    # Run the chatbot
    run_chat(chain)

if __name__ == "__main__":
    main()
