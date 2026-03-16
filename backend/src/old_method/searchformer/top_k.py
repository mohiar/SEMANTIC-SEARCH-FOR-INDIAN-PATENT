# 🚀 Imports
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from datetime import datetime

# 🚀 1. Load Fine-Tuned Model
model = SentenceTransformer('../fine_tuned_patent_model')

# 🚀 2. Load Pre-computed Fine-Tuned Embeddings
doc_embeddings = np.load('../fine_tuned_embeddings.npy')
doc_embeddings = normalize(doc_embeddings)

# 🚀 3. Load Patent Data
with open('patent_search_results.json', 'r', encoding='utf-8') as f:
    patent_data = json.load(f)

# 🚀 4. Search Loop with Logging
log = []

while True:
    query = input("\n🔎 Enter your search query (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    # 🚀 Encode & Normalize Query
    query_embedding = model.encode(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # 🚀 Similarity Calculation
    similarities = np.dot(doc_embeddings, query_embedding)
    top_k = int(input("enter top k:"))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 🚀 Display & Collect Results
    print(f"\n🎯 Top-{top_k} Results for Query: '{query}'\n")
    results = []
    for idx in top_indices:
        title = patent_data[idx].get('title', 'N/A')
        abstract = patent_data[idx].get('abstract', 'N/A')
        app_number = patent_data[idx].get('application_number', 'N/A')
        score = float(round(similarities[idx], 4))  # ✅ Ensure it's a native float


        results.append({
            'application_number': str(app_number),
            'title': str(title),
            'abstract': str(abstract),
            'similarity_score': score
        })

    # ✅ Store with timestamp
    log.append({
        'timestamp': datetime.now().isoformat(),
        'query': str(query),
        'results': results
    })

# 🚀 Final Save to Log File
try:
    with open('search_log.json', 'w', encoding='utf-8') as log_file:
        json.dump(log, log_file, ensure_ascii=False, indent=4)
    print("📝 Search log saved to 'search_log.json'")
except Exception as e:
    print(f"❌ Error writing to log: {e}")
