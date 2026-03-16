#!/usr/bin/env python3
"""
BM25 + Semantic Search Re-ranking Module
=========================================
Combines BM25 keyword search with Sentence-BERT semantic re-ranking
for improved patent and paper search results.
"""

import re
import json
import csv
import numpy as np
import logging
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BM25SemanticSearch:
    """
    Hybrid search combining BM25 keyword matching with semantic embeddings.
    """
    
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize the BM25 semantic search engine.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        logger.info(f"Initializing BM25SemanticSearch with model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("✅ Sentence-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
        
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'fig\.\d+', '', text)
        text = re.sub(r'figure\s\d+', '', text)
        return text.strip()
    
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Tokenize text into words"""
        return BM25SemanticSearch.clean_text(text).split()
    
    def build_index(self, documents: List[Dict[str, str]]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries with 'abstract' field
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        try:
            # Extract abstracts
            self.corpus = [doc.get('abstract', '') for doc in documents]
            
            # Tokenize corpus
            self.tokenized_corpus = [self.tokenize_text(doc) for doc in self.corpus]
            
            # Build BM25 index
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            logger.info(f"✅ BM25 index built successfully with {len(self.corpus)} documents")
            
        except Exception as e:
            logger.error(f"❌ Error building index: {e}")
            raise
    
    def search(self, query: str, documents: List[Dict[str, str]], 
               top_k: int = 10, similarity_threshold: float = 0.45) -> List[Dict[str, str]]:
        """
        Perform hybrid BM25 + semantic search.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top BM25 candidates to re-rank semantically
            similarity_threshold: Minimum cosine similarity threshold (0-1)
        
        Returns:
            List of ranked results sorted by semantic similarity
        """
        logger.info(f"Searching for: '{query}'")
        
        try:
            # Build index if not already built
            if not self.bm25:
                self.build_index(documents)
            
            # Step 1: BM25 Search
            logger.info(f"Step 1: Running BM25 search (top_k={top_k})...")
            tokenized_query = self.tokenize_text(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k BM25 candidates
            top_n_indices = np.argsort(bm25_scores)[::-1][:min(top_k, len(documents))]
            candidate_docs = [documents[i] for i in top_n_indices]
            
            logger.info(f"✅ Retrieved {len(candidate_docs)} BM25 candidates")
            
            # Step 2: Semantic Re-ranking
            logger.info("Step 2: Semantic re-ranking with Sentence-BERT...")
            
            # Generate embeddings
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            doc_abstracts = [doc.get('abstract', '') for doc in candidate_docs]
            doc_embeddings = self.model.encode(doc_abstracts, convert_to_tensor=False)
            
            # Calculate cosine similarity
            cosine_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Step 3: Filtering and Re-ranking
            logger.info(f"Step 3: Filtering by threshold ({similarity_threshold})...")
            
            final_results = []
            for idx, (original_idx, score) in enumerate(zip(top_n_indices, cosine_scores)):
                doc = documents[original_idx]
                
                if score >= similarity_threshold:
                    result = {
                        "title": doc.get('title', ''),
                        "abstract": doc.get('abstract', ''),
                        "source": doc.get('source', 'Unknown'),
                        "similarity_score": round(float(score), 4),
                        "url": doc.get('url', ''),
                        "application_number": doc.get('application_number', '')
                    }
                    final_results.append(result)
            
            # Sort by similarity score (descending)
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"✅ Search completed. Found {len(final_results)} results above threshold")
            
            return final_results
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            raise
    
    def search_by_indices(self, query: str, top_k: int = 10) -> List[int]:
        """
        Perform search and return only the indices of top results.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of indices in the original corpus
        """
        if not self.bm25:
            logger.error("❌ Index not built. Call build_index() first.")
            return []
        
        try:
            tokenized_query = self.tokenize_text(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get embeddings
            query_embedding = self.model.encode(query)
            doc_embeddings = self.model.encode(self.corpus)
            
            # Calculate similarity
            cosine_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            # Get top-k by cosine similarity
            top_indices = np.argsort(cosine_scores)[::-1][:min(top_k, len(self.corpus))]
            
            return list(top_indices)
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return []


# ============================================================================
# STANDALONE USAGE (for testing)
# ============================================================================

def main():
    """Standalone execution for testing"""
    import sys
    
    print("\n" + "="*70)
    print("🔍 BM25 SEMANTIC SEARCH ENGINE")
    print("="*70 + "\n")
    
    # Load documents
    try:
        with open('patent_search_results.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"✅ Loaded {len(documents)} documents from patent_search_results.json")
    except FileNotFoundError:
        logger.error("❌ File 'patent_search_results.json' not found")
        return
    except Exception as e:
        logger.error(f"❌ Error loading documents: {e}")
        return
    
    # Initialize searcher
    try:
        searcher = BM25SemanticSearch()
    except Exception as e:
        logger.error(f"❌ Failed to initialize searcher: {e}")
        return
    
    # Build index
    try:
        searcher.build_index(documents)
    except Exception as e:
        logger.error(f"❌ Failed to build index: {e}")
        return
    
    # Interactive search loop
    while True:
        print("\n" + "-"*70)
        query = input("\n🔎 Enter search query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not query:
            print("❌ Query cannot be empty")
            continue
        
        try:
            top_k = input("📋 Top-N BM25 candidates to re-rank (default 20): ").strip()
            top_k = int(top_k) if top_k else 20
            
            threshold = input("🎯 Similarity threshold (default 0.45): ").strip()
            threshold = float(threshold) if threshold else 0.45
            
            # Perform search
            results = searcher.search(query, documents, top_k=top_k, similarity_threshold=threshold)
            
            # Display results
            print("\n" + "="*70)
            print(f"✅ Found {len(results)} results\n")
            
            for i, result in enumerate(results[:10], 1):
                print(f"{i}. {result['title'][:60]}...")
                print(f"   Score: {result['similarity_score']} | Source: {result['source']}")
                print()
            
            # Save results
            output_file = f"search_results_{query.replace(' ', '_')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_file}")
        
        except ValueError:
            print("❌ Invalid input for top_k or threshold")
        except Exception as e:
            logger.error(f"❌ Search error: {e}")


if __name__ == "__main__":
    main()