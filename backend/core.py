import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Initialize the embeddings(same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialize the vector store(same as ingestion.py)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings,
)

#Initialize the chat model(same as ingestion.py)
model = init_chat_model(
    model="gpt-5.2",
    model_provider="openai",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about Langchain."""
    retreived_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retreived_docs
        )

    # Return both Serialized content and raw documents
    return serialized, retreived_docs

def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's qestion.

    Returns:
        Dictionary containing:
            - answer: The generated answer.
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that can answer questions about Langchain documentation."
        "You have access to a tool that can retrieve relevant documentation."
        "Use this tool to find the most relevant documentation before answering questions."
        "Use the tool to find relevant information before answering questions."
        "If you don't know the answer, say so."
        "If you know the answer, answer the question."
    )
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="whar are deep agents?")
    print(result)    





