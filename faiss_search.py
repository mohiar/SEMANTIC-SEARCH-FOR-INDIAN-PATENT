import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Load entries from the JSON file exported from the scraper
with open('scholar_papers.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)

# Get all abstracts, filtering out entries that might not have one
abstracts = [entry.get("abstract", "") for entry in entries if entry.get("abstract")]
# Keep track of the original entries that have abstracts
valid_entries = [entry for entry in entries if entry.get("abstract")]


# Encode abstracts to normalized vectors
abstract_embeddings = model.encode(abstracts, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index for cosine similarity (using inner product on normalized vectors)
dimension = abstract_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(abstract_embeddings)

# Multiple enriched query variations
queries = [
  "How AI is used in smart beekeeping systems powered by solar energy",
  "Solar-powered IoT devices for real-time hive monitoring in apiculture",
  "AI algorithms for detecting bee colony health and behavior patterns",
  "Sustainable beekeeping using renewable energy and machine learning",
  "Energy-efficient smart hive systems with predictive analytics for beekeeping"
]

# Encode and average the query embeddings
query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
query_embedding = np.mean(query_embeddings, axis=0).reshape(1, -1)

# Search top-K similar entries
top_k = 15  # Limit to top 15 for stricter filtering
distances, indices = index.search(query_embedding, top_k)

# Define a similarity threshold
threshold = 0.5  # Stricter filtering for relevance

# Optional keyword-based beekeeping filter
# beekeeping_keywords =["solar", "powered" ,"ai", "beekeeping" ,"system"]

# def is_beekeeping_related(text):
#     return any(word in text.lower() for word in beekeeping_keywords)

# Collect relevant entries
results = []
print(f"Top {top_k} results from FAISS (before filtering):")
for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
    entry = valid_entries[idx]
    print(f"  - Result {i+1}: Score={score:.3f}, Title={entry.get('title')}")

for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
    entry = valid_entries[idx] # Use the filtered list of valid entries
    abstract_text = entry["abstract"]
    # if score >= threshold and is_beekeeping_related(abstract_text) :
    if score >= threshold:
        results.append({
            "Title": entry.get("title"),
            "URL": entry.get("paper_url"), # Changed from Application Number
            "Abstract": abstract_text,
            "Similarity": round(float(score), 3)
        })

# Save results to a new JSON file
json_filename = "scholar_search_results_relevant_beekeeping.json"
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Filtered beekeeping-relevant results saved to {json_filename}")