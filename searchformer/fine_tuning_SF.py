# 🚀 Fine-tune Semantic Search Model (Mini-SEARCHFORMER) from Patent Abstracts

# 📚 Import Libraries
import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 🚀 1. Load Patent Abstracts
with open('patent_search_results.json', 'r', encoding='utf-8') as f:
    patent_data = json.load(f)

abstracts = [entry['abstract'] for entry in patent_data]
titles = [entry['title'] for entry in patent_data]

# 🚀 2. Prepare Positive and Negative Pairs
train_examples = []


# Strategy:
# - Randomly pick 1 positive (same topic / similar title keywords)
# - Randomly pick 1 negative (different topic)

# Small helper function
def is_similar(title1, title2):
    title1 = title1.lower()
    title2 = title2.lower()
    common_words = set(title1.split()) & set(title2.split())
    return len(common_words) >= 2  # If at least 2 words match → consider as positive


print("[INFO] Creating sentence pairs...")
for i in range(len(abstracts)):
    anchor = abstracts[i]
    anchor_title = titles[i]

    # Find a positive sample (with similar title keywords)
    positive = None
    for j in range(len(abstracts)):
        if i != j and is_similar(anchor_title, titles[j]):
            positive = abstracts[j]
            break

    # If no good positive found, just pick next document
    if not positive:
        positive = abstracts[(i + 1) % len(abstracts)]

    # Random negative sample
    negative_idx = random.choice([k for k in range(len(abstracts)) if k != i])
    negative = abstracts[negative_idx]

    # Add positive example (label 1)
    train_examples.append(InputExample(texts=[anchor, positive], label=1.0))
    # Add negative example (label 0)
    train_examples.append(InputExample(texts=[anchor, negative], label=0.0))

print(f"[INFO] Created {len(train_examples)} training pairs.")

# 🚀 3. Load Pre-trained Base Model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Or load PatentSBERTa if preferred

# 🚀 4. Prepare DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 🚀 5. Set Loss Function
train_loss = losses.CosineSimilarityLoss(model=model)

# 🚀 6. Fine-Tune Model
print("[INFO] Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,  # 2-3 epochs enough for small data
    warmup_steps=100
)

# 🚀 7. Save Your Fine-tuned Model
output_path = '../fine_tuned_patent_model'
model.save(output_path)

print(f"[INFO] Fine-tuned model saved to '{output_path}' successfully!")

# 🚀 8. Load Later Like:
# model = SentenceTransformer('fine_tuned_patent_model')
