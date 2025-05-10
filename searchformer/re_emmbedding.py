# 🚀 Re-embed patent abstracts using Fine-Tuned Patent Model

# 📚 Import Libraries
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 🚀 1. Load Fine-Tuned Model
model = SentenceTransformer('../fine_tuned_patent_model')

# 🚀 2. Load Patent Abstracts
with open('patent_search_results.json', 'r', encoding='utf-8') as f:
    patent_data = json.load(f)

abstracts = [entry['abstract'] for entry in patent_data]

# 🚀 3. Generate Embeddings
print("[INFO] Generating new embeddings with fine-tuned model...")
embeddings = []

for abstract in tqdm(abstracts, desc="Encoding Abstracts"):
    emb = model.encode(abstract)
    embeddings.append(emb)

embeddings = np.array(embeddings)

# 🚀 4. Save the Embeddings
np.save('../fine_tuned_embeddings.npy', embeddings)

print(f"\n✅ Fine-tuned embeddings saved successfully as 'fine_tuned_embeddings.npy'!")
