#!/usr/bin/env python3
"""
invert_with_sentence_transformers.py

Attempt to approximate/recover text from a target embedding by local search (no external DB).
- Tries to import generate_embeddings from user_embeddings.py (if you have your function there).
- Otherwise uses SentenceTransformer('all-MiniLM-L6-v2') locally.
- Two search strategies:
    * evolutionary_search: population + mutation
    * greedy_local_search: hill-climb / random restarts

Usage:
  python invert_with_sentence_transformers.py --target-text "I like pizza."
  python invert_with_sentence_transformers.py --target-emb target_embedding.json

Requires:
  - sentence-transformers (if you don't provide user_embeddings.py)
  - numpy
"""
import argparse
import json
import os
import random
import sys
import time
from typing import List, Callable

import numpy as np

# Try to import user-provided generate_embeddings(texts) to preserve exact behaviour
get_embeddings: Callable[[List[str]], List[List[float]]]
try:
    from user_embeddings import generate_embeddings as get_embeddings  # type: ignore
    print("Using generate_embeddings from user_embeddings.py")
except Exception:
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL_NAME = "all-MiniLM-L6-v2"
        _st_model = SentenceTransformer(_MODEL_NAME)
        print(f"Using local SentenceTransformer('{_MODEL_NAME}') for embeddings.")

        def get_embeddings(texts: List[str]) -> List[List[float]]:
            # returns list-of-lists
            return _st_model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        raise RuntimeError(
            "No generate_embeddings found and SentenceTransformer not available. "
            "Either install sentence-transformers or provide user_embeddings.generate_embeddings(texts)."
        ) from e

def cos_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

# Mutation utilities
ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-'\"?!")
WORD_SEEDS = ["the", "a", "I", "you", "it", "is", "are", "like", "love", "enjoy", "this", "example", "test", "model"]

def mutate_charwise(s: str, p_ins=0.12, p_del=0.08, p_sub=0.18) -> str:
    s_list = list(s)
    # deletion
    if len(s_list) > 0 and random.random() < p_del:
        idx = random.randrange(len(s_list))
        del s_list[idx]
    # substitution
    if len(s_list) > 0 and random.random() < p_sub:
        idx = random.randrange(len(s_list))
        s_list[idx] = random.choice(ALPHABET)
    # insertion
    if random.random() < p_ins:
        idx = random.randrange(len(s_list) + 1)
        s_list.insert(idx, random.choice(ALPHABET))
    return "".join(s_list)

def mutate_wordwise(s: str) -> str:
    words = s.split()
    if not words:
        # seed from common words
        return random.choice(WORD_SEEDS)
    op = random.random()
    if op < 0.33:  # replace a word
        i = random.randrange(len(words))
        words[i] = random.choice(WORD_SEEDS + [random.choice(words)])
    elif op < 0.66 and len(words) < 40:  # insert
        i = random.randrange(len(words) + 1)
        words.insert(i, random.choice(WORD_SEEDS))
    else:  # delete
        if len(words) > 1:
            i = random.randrange(len(words))
            del words[i]
    return " ".join(words)

# Evolutionary search
def evolutionary_search(target_emb: List[float],
                        initial_population: List[str] = None,
                        pop_size: int = 60,
                        generations: int = 120,
                        elite_frac: float = 0.2) -> List[tuple]:
    if initial_population is None:
        initial_population = [
            "A short sentence.",
            "I like this.",
            "This is interesting.",
            "Machine learning example.",
            "I enjoy coding.",
            "The quick brown fox jumps over the lazy dog.",
            "An example sentence.",
            "I love pizza."
        ]

    # create initial population by mutating seeds
    population = []
    while len(population) < pop_size:
        base = random.choice(initial_population)
        # apply multiple random mutations to diversify
        s = base
        for _ in range(random.randint(1, 4)):
            if random.random() < 0.6:
                s = mutate_charwise(s)
            else:
                s = mutate_wordwise(s)
        population.append(s)

    best_overall = []
    for gen in range(generations):
        # batch-embed
        emb_list = get_embeddings(population)
        scores = [cos_sim(target_emb, e) for e in emb_list]
        scored = list(zip(scores, population))
        scored.sort(reverse=True, key=lambda x: x[0])
        n_elite = max(1, int(pop_size * elite_frac))
        elites = [s for _, s in scored[:n_elite]]
        best_score, best_text = scored[0]
        if gen % 10 == 0 or gen == generations - 1:
            print(f"[Evo] Gen {gen+1}/{generations} best={best_score:.4f} text={best_text!r}")
        best_overall.append(scored[0])

        # Terminate early if extremely close
        if best_score > 0.995:
            print("[Evo] early stop: almost-perfect similarity")
            break

        # create next generation
        next_pop = elites.copy()
        while len(next_pop) < pop_size:
            parent = random.choice(elites)
            child = parent
            # mix charwise and wordwise mutations
            if random.random() < 0.6:
                child = mutate_charwise(child,
                                       p_ins=0.12 + random.random()*0.2,
                                       p_del=0.06 + random.random()*0.2,
                                       p_sub=0.12 + random.random()*0.3)
            else:
                child = mutate_wordwise(child)
            # occasional crossover
            if random.random() < 0.12:
                other = random.choice(elites)
                split = random.randrange(1, max(2, min(len(child), len(other))))
                child = child[:split] + other[split:]
            next_pop.append(child)
        population = next_pop

    best_overall.sort(reverse=True, key=lambda x: x[0])
    return best_overall[:30]

# Greedy local/hill-climb with restarts
def greedy_local_search(target_emb: List[float],
                        restarts: int = 12,
                        iters_per_restart: int = 200) -> List[tuple]:
    results = []
    for r in range(restarts):
        # seed: short template or random word seed
        seed = random.choice(WORD_SEEDS + ["This is an example.", "I like this."])
        cur = seed
        cur_emb = get_embeddings([cur])[0]
        cur_score = cos_sim(target_emb, cur_emb)
        if r % 4 == 0:
            print(f"[Greedy] restart {r+1}/{restarts} seed={seed!r} score={cur_score:.4f}")
        for it in range(iters_per_restart):
            # propose candidate
            if random.random() < 0.6:
                cand = mutate_charwise(cur)
            else:
                cand = mutate_wordwise(cur)
            cand_emb = get_embeddings([cand])[0]
            cand_score = cos_sim(target_emb, cand_emb)
            if cand_score > cur_score:
                cur, cur_score = cand, cand_score
            # occasional random jump
            if random.random() < 0.01:
                cur = random.choice(WORD_SEEDS)
                cur_score = cos_sim(target_emb, get_embeddings([cur])[0])
        results.append((cur_score, cur))
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:30]

def load_embedding_from_file(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Embedding file must contain a JSON list of floats.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Invert a sentence-transformers embedding by search.")
    parser.add_argument("--target-text", type=str, help="Compute target embedding from this text (quick demo).")
    parser.add_argument("--target-emb", type=str, help="Path to JSON file containing target embedding (list of floats).")
    parser.add_argument("--mode", choices=["evo", "greedy", "both"], default="both")
    parser.add_argument("--seed-candidates", type=str, nargs="*", default=None,
                        help="Optional list of seed candidate strings to bias search.")
    args = parser.parse_args()

    if not args.target_text and not args.target_emb:
        print("Provide --target-text or --target-emb. Example:\n  python invert_with_sentence_transformers.py --target-text \"I love pizza.\"")
        sys.exit(1)

    if args.target_text:
        print("Computing target embedding for provided text...")
        target_emb = get_embeddings([args.target_text])[0]
    else:
        target_emb = load_embedding_from_file(args.target_emb)
        print(f"Loaded target embedding (len={len(target_emb)})")

    # optional seeds
    seeds = args.seed_candidates if args.seed_candidates else None

    all_results = []

    if args.mode in ("evo", "both"):
        print("Starting evolutionary search...")
        evo_res = evolutionary_search(target_emb, initial_population=seeds, pop_size=60, generations=120)
        all_results.extend(evo_res)

    if args.mode in ("greedy", "both"):
        print("Starting greedy local search...")
        greedy_res = greedy_local_search(target_emb, restarts=12, iters_per_restart=200)
        all_results.extend(greedy_res)

    # deduplicate and sort
    uniq = {}
    for score, txt in all_results:
        if txt not in uniq or uniq[txt] < score:
            uniq[txt] = score
    final = sorted([(s, t) for t, s in uniq.items()], reverse=True, key=lambda x: x[0])

    print("\nTop candidates:")
    for score, txt in final[:30]:
        print(f"{score:.4f}  {txt}")

if __name__ == "__main__":
    main()