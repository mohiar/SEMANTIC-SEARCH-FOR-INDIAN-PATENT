# 📚 Import Libraries
import re
import json
import csv
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# 🚀 1. Preprocessing Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'fig\.\d+', '', text)
    text = re.sub(r'figure\s\d+', '', text)
    return text

def tokenize_text(text):
    return clean_text(text).split()

# 🚀 2. Load Patent Abstracts
with open('patent_search_results.json', 'r', encoding='utf-8') as f:
    patent_data = json.load(f)


corpus = [doc['abstract'] for doc in patent_data]

# 🚀 3. Build BM25 Index
print("[INFO] Building BM25 index...")
tokenized_corpus = [tokenize_text(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 🚀 4. Load Semantic Embedding Model
print("[INFO] Loading Sentence-BERT model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')   # Or load PatentSBERTa if you have it

# 🚀 5. Take User Inputs
user_query = input("\n🔎 Enter your search query: ")
top_k = int(input("\n📋 Enter how many Top-N BM25 results to consider for re-ranking (e.g., 20, 50): "))
similarity_threshold = float(input("\n🎯 Enter similarity threshold (recommended 0.45): "))

# 🚀 6. BM25 Search
tokenized_query = tokenize_text(user_query)
bm25_scores = bm25.get_scores(tokenized_query)
top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
candidate_docs = [patent_data[i] for i in top_n_indices]

print(f"\n[INFO] Retrieved Top-{top_k} BM25 candidates.")

# 🚀 7. Semantic Embedding
print("[INFO] Generating embeddings...")
query_embedding = model.encode(user_query)
doc_embeddings = model.encode([doc['abstract'] for doc in candidate_docs])

# 🚀 8. Re-ranking Using Cosine Similarity
print("[INFO] Re-ranking candidates...")
cosine_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
re_ranked_indices = np.argsort(cosine_scores)[::-1]

# 🚀 9. Filtering by Threshold
final_results = []
for idx in re_ranked_indices:
    if cosine_scores[idx] >= similarity_threshold:
        final_results.append({
            "application_number": candidate_docs[idx].get('application_number', ''),
            "title": candidate_docs[idx]['title'],
            "abstract": candidate_docs[idx]['abstract'],
            "similarity_score": round(float(cosine_scores[idx]), 3)
        })

print(f"\n✅ {len(final_results)} results found above threshold {similarity_threshold}.")

# 🚀 10. Save Results
output_json = 'semantic_search_results.json'
output_csv = 'semantic_search_results.csv'

# Save to JSON
with open(output_json, 'w') as f:
    json.dump(final_results, f, indent=4)
print(f"[INFO] Results saved to '{output_json}'.")

# Save to CSV (Excel Friendly)
keys = final_results[0].keys() if final_results else []
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    dict_writer = csv.DictWriter(f, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(final_results)
print(f"[INFO] Results also saved to '{output_csv}'.")

# 🚀 11. Show Top Results on Console
print("\n🔎 Top Results:")
for result in final_results[:5]:  # Show Top-5
    print(f"\nTitle: {result['title']}\nSimilarity Score: {result['similarity_score']}\n")
