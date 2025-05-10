import json
from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Load entries from JSON file
with open('../searchformer/patent_search_results.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)

# Your target concept
query = "A portable battery that allows you to charge electronic devices like smartphones, tablets, and laptops on the go."

# Encode query and abstracts
query_embedding = model.encode(query, convert_to_tensor=True)
abstracts = [entry["abstract"] for entry in entries]
abstract_embeddings = model.encode(abstracts, convert_to_tensor=True)

# Cosine similarity
cosine_scores = util.cos_sim(query_embedding, abstract_embeddings).squeeze()

# Set similarity threshold
threshold = 0.45

# Filter matching abstracts
matching_entries = [
    {
        "title": entries[i]["title"],
        "application_number": entries[i]["application_number"],
        "abstract":entries[i]["abstract"],
        "similarity": float(cosine_scores[i])
    }
    for i in range(len(entries)) if cosine_scores[i] >= threshold
]

# Print matching entries
for entry in matching_entries:
    print(f"\nTitle: {entry['title']}")
    print(f"Application Number: {entry['application_number']}")
    print(f"Abstract : {entry['abstract']}")
    print(f"Similarity Score: {entry['similarity']:.3f}")
    print("-"*170)
