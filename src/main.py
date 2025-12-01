from embeddings import generate_embeddings
from chromadb_store import store_embeddings, extract_embeddings
# from cyborgdb_store import store_embeddings_secure, extract_embeddings_secure
# from attack import run_attack

with open("data/sensitive_samples.txt") as f:
    texts = f.readlines()

texts = [t.strip() for t in texts]

print("\n[1] Generating embeddings...")
embeddings = generate_embeddings(texts)
# print(f"Generated {embeddings} embeddings.")

print("\n[2] Storing in ChromaDB (PLAIN)...")
store_embeddings(texts, embeddings)

print("\n[3] Extracting from ChromaDB...")
leaked_vectors = extract_embeddings()
print(f"Extracted {leaked_vectors} embeddings.")

# print("\n[4] Running vec2text attack on ChromaDB...")
# recovered_plain = run_attack(leaked_vectors)

# print("\n[RESULT] Recovered from ChromaDB:")
# for r in recovered_plain:
    # print(" -", r)

# print("\n[5] Storing in CyborgDB (ENCRYPTED)...")
# store_embeddings_secure(embeddings)

# print("\n[6] Extracting from CyborgDB...")
# secure_vectors = extract_embeddings_secure()

# print("\n[7] Running vec2text attack on CyborgDB...")
# recovered_secure = run_attack(secure_vectors)

# print("\n[RESULT] Recovered from CyborgDB:")
# for r in recovered_secure:
#     print(" -", r)
