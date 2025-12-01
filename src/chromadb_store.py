import chromadb

client = chromadb.Client()
collection = client.create_collection("leak_test")

def store_embeddings(texts, embeddings):
    for i, (t, e) in enumerate(zip(texts, embeddings)):
        collection.add(documents=[t], embeddings=[e], ids=[str(i)])

def extract_embeddings():
    results = collection.get(include=["embeddings"])
    return results["embeddings"]
