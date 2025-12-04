"""
Demo: extract embeddings from ChromaDB and reconstruct original text using LangChain + OpenAI.

Usage:
  1. Set OPENAI_API_KEY in env.
  2. pip install -r requirements.txt
  3. python demo_reconstruct.py

Behavior:
  - Creates an in-memory Chroma collection
  - Adds sample documents, computed via OpenAIEmbeddings (via LangChain)
  - Reads the stored embeddings for a document id
  - Performs a similarity search by vector to retrieve the original document(s)
  - Optionally calls an OpenAI LLM (via LangChain) to "reconstruct" the text from retrieved candidates
"""
from dotenv import load_dotenv
import os
import sys
import uuid
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_huggingface import embeddings
from langchain_huggingface.chat_models import ChatHuggingFace

# Load environment variables
load_dotenv()

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment")


# ------------ Configuration ------------
# You can override these via env vars if needed
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = "x-ai/grok-4.1-fast:free"
CHROMA_PERSIST_DIR = None  # set to a path to persist across runs, or None for in-memory
# ---------------------------------------





def create_sample_documents() -> List[str]:
    return [
        "The quick brown fox jumped over the lazy dog. This sentence is used for demo purposes.",
        "LangChain and Chroma make it easy to store embeddings and perform similarity search.",
        "OpenAI embeddings convert text into vectors which we store in Chroma. Later we can query by vector.",
        "This is a short note about using embeddings and reconstructing text via nearest-neighbor retrieval."
    ]


def init_chroma_client():
    # Use in-memory or persistent Chroma depending on CHROMA_PERSIST_DIR
    if CHROMA_PERSIST_DIR:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
    else:
        client = chromadb.Client()
    return client


async def main():
    # require_api_key()
    docs = create_sample_documents()
    ids = [str(uuid.uuid4()) for _ in docs]

    print("Initializing LangChain HuggingFaceEmbeddings embeddings (model=%s) ..." % EMBEDDING_MODEL)
   
    vectors = await embeddings.HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL).aembed_documents(docs)
    # print("Initialized.",vectors)

    client = init_chroma_client()
    collection_name = "demo_reconstruct_collection"

    # Create (or get) collection
    try:
        collection = client.create_collection(name=collection_name)
        print(f"Created collection '{collection_name}'.")
    except Exception:
        collection = client.get_collection(name=collection_name)
        print(f"Re-using collection '{collection_name}'.")

    # Compute embeddings via LangChain
    print("Computing embeddings for sample documents...")
    # vectors = embeddings.embed_documents(docs)

    # # Add documents + embeddings into Chroma
    print("Adding documents to ChromaDB with embeddings...")
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=[{"source": "demo"} for _ in docs],
        embeddings=vectors,
    )
    if CHROMA_PERSIST_DIR:
        client.persist()

    # # Pick a target id to "reconstruct"
    target_index = 0
    target_id = ids[target_index]
    print("\nTarget document id to reconstruct:", target_id)

    # # Retrieve the stored embedding for this id from Chroma
    print("\nRetrieving stored embedding from ChromaDB for the id...")
    get_result = collection.get(ids=[target_id], include=["embeddings", "documents", "metadatas"])
    stored_embeddings = get_result.get("embeddings", [])
    stored_documents = get_result.get("documents", [])
    if not stored_embeddings.any():
        print("No embedding found for id", target_id)
        return

    stored_embedding = stored_embeddings[0]  # vector
    stored_doc_text = stored_documents[0]
    print("Stored document text (ground truth):")
    print("----")
    print(stored_doc_text)
    print("----")

    # Use the vector to do a similarity query (nearest neighbors)
    print("\nPerforming similarity search by vector to reconstruct (nearest neighbors)...")
    # Note: API naming: query(query_embeddings=...) or query(query_texts=...). We pass vector directly.
    query_result = collection.query(
        query_embeddings=[stored_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    # Display nearest neighbors returned by Chroma
    results_docs = query_result["documents"][0]
    results_ids = query_result["ids"][0]
    results_distances = query_result["distances"][0]
    print("\nNearest neighbors (by Chroma):")
    for i, (rid, doc_text, dist) in enumerate(zip(results_ids, results_docs, results_distances), start=1):
        print(f"{i}. id={rid}  distance={dist:.6f}")
        print(doc_text)
        print("----")

    # # Optional: Use an LLM (LangChain ChatOpenAI) to 'reconstruct' or refine the text based on retrieved neighbors.
    # # This is not an exact inverted embedding; it uses retrieved candidate texts to ask the LLM to produce the most
    # # likely reconstruction of the original content.
    print("\nUsing OpenAI LLM (via LangChain) to reconstruct/refine from the retrieved candidates...")
    chat = ChatHuggingFace(model_name=LLM_MODEL, openrouter_api_key=OPENROUTER_API_KEY, temperature=0.7, verbose=True)

    # # Build a prompt that includes the retrieved candidates and asks the LLM to produce the most likely original.
    prompt = (
        "You are given a set of candidate text snippets returned by vector similarity for a single original document.\n"
        "Use the snippets to reconstruct the original document as faithfully as possible. If the original is present in a snippet, reproduce it.\n\n"
        "Candidates:\n"
    )
    for idx, txt in enumerate(results_docs, start=1):
        prompt += f"\n--- Candidate {idx} ---\n{txt}\n"

    prompt += (
        "\n\nNow produce the reconstructed text (only the reconstructed text, no commentary)."
        "\nIf you are uncertain about missing parts, try to produce the most plausible reconstruction."
    )

    # response = chat([human(content=prompt)])
    # reconstruction = response.content.strip()
    # print("\nReconstructed text from LLM:")
    # print("----")
    # print(reconstruction)
    # print("----")

    # # Compare (quick manual note)
    # print("\nGround-truth vs reconstructed (manual comparison):")
    # print("GROUND TRUTH:")
    # print(stored_doc_text)
    # print("\nRECONSTRUCTED:")
    # print(reconstruction)

    # print("\nDemo complete.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())