#!/usr/bin/env python3
"""
FastAPI Application for Semantic Search Pipeline
================================================
Handles user requests asynchronously with background task processing.
- Google Scholar scraping
- Patent Office scraping
- BM25 semantic search ranking
- Results stored and retrieved via API
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import logging
import uuid
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Patent Search API",
    description="Search for patents and academic papers using semantic search",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """User search request model"""
    query: str
    include_patents: bool = True
    include_papers: bool = True
    top_k: int = 10
    

class SearchResult(BaseModel):
    """Individual search result"""
    title: str
    abstract: str
    source: str
    url: Optional[str] = None
    application_number: Optional[str] = None
    authors: Optional[str] = None
    similarity_score: Optional[float] = None


class SearchResponse(BaseModel):
    """API response for search requests"""
    request_id: str
    query: str
    status: str  # "processing", "completed", "failed"
    total_results: int
    results: List[SearchResult] = []
    timestamp: str
    error_message: Optional[str] = None


# ============================================================================
# IN-MEMORY STORAGE FOR SEARCH RESULTS
# ============================================================================

search_cache = {}
RESULTS_DIR = "search_results"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_search_results(request_id: str, response: SearchResponse):
    """Save search results to a JSON file"""
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response.dict(), f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Saved results for request {request_id}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def load_json_data(filepath):
    """Load data from a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format: {filepath}")
        return []


def combine_sources(google_scholar_data, patent_data):
    """Combine results from both sources"""
    combined = []
    
    # Add Google Scholar papers
    for paper in google_scholar_data:
        combined.append({
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("paper_url", ""),
            "source": "Google Scholar",
            "authors": paper.get("authors", "")
        })
    
    # Add Patents
    for patent in patent_data:
        combined.append({
            "title": patent.get("title", ""),
            "abstract": patent.get("abstract", ""),
            "url": patent.get("patent_url", ""),
            "application_number": patent.get("application_number", ""),
            "source": "Indian Patent Office",
        })
    
    logger.info(f"📊 Combined {len(google_scholar_data)} papers + {len(patent_data)} patents = {len(combined)} total")
    return combined


def apply_bm25_ranking(combined_data, query, top_k=10):
    """Apply BM25 semantic search ranking"""
    try:
        from backend.src.bm25.semantic_search_bm25 import BM25SemanticSearch
        
        logger.info(f"🔍 Applying BM25 ranking for query: '{query}'")
        
        searcher = BM25SemanticSearch()
        documents = [
            f"{item.get('title', '')} {item.get('abstract', '')}"
            for item in combined_data
        ]
        
        ranked_indices = searcher.search(query, documents, top_k=min(top_k, len(combined_data)))
        ranked_results = [combined_data[idx] for idx in ranked_indices]
        
        logger.info(f"✅ BM25 ranking completed")
        return ranked_results
        
    except ImportError:
        logger.warning("⚠️  BM25 module not found. Returning combined results without ranking.")
        return combined_data[:top_k]
    except Exception as e:
        logger.error(f"Error during BM25 ranking: {e}")
        return combined_data[:top_k]


def process_search_request(request_id: str, query: str, include_patents: bool, 
                          include_papers: bool, top_k: int):
    """Background task to process search request"""
    logger.info(f"🔄 Processing search request {request_id}: '{query}'")
    
    try:
        # Initialize response
        response = SearchResponse(
            request_id=request_id,
            query=query,
            status="processing",
            total_results=0,
            timestamp=datetime.now().isoformat()
        )
        
        # Load data from sources
        google_scholar_data = []
        patent_data = []
        
        if include_papers:
            google_scholar_data = load_json_data("google_scholar_results.json")
            logger.info(f"✅ Loaded {len(google_scholar_data)} Google Scholar papers")
        
        if include_patents:
            patent_data = load_json_data("patent_search_results.json")
            logger.info(f"✅ Loaded {len(patent_data)} patents")
        
        if not google_scholar_data and not patent_data:
            response.status = "failed"
            response.error_message = "No data available. Please run the scrapers first."
            search_cache[request_id] = response
            save_search_results(request_id, response)
            return
        
        # Combine data
        combined_data = combine_sources(google_scholar_data, patent_data)
        
        # Apply BM25 ranking
        ranked_results = apply_bm25_ranking(combined_data, query, top_k)
        
        # Format results
        results = []
        for result in ranked_results:
            results.append(SearchResult(
                title=result.get("title", ""),
                abstract=result.get("abstract", ""),
                source=result.get("source", ""),
                url=result.get("url"),
                application_number=result.get("application_number"),
                authors=result.get("authors"),
                similarity_score=result.get("similarity_score")
            ))
        
        # Update response
        response.status = "completed"
        response.total_results = len(results)
        response.results = results
        
        # Cache and save results
        search_cache[request_id] = response
        save_search_results(request_id, response)
        
        logger.info(f"✅ Search request {request_id} completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error processing search request {request_id}: {e}")
        response = SearchResponse(
            request_id=request_id,
            query=query,
            status="failed",
            total_results=0,
            timestamp=datetime.now().isoformat(),
            error_message=str(e)
        )
        search_cache[request_id] = response
        save_search_results(request_id, response)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "🚀 Semantic Patent Search API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.post("/search", response_model=dict, tags=["Search"])
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Submit a search query for processing.
    
    The search will be processed in the background and results can be retrieved
    using the returned request_id.
    
    Args:
        request: SearchRequest containing query and options
        background_tasks: FastAPI background tasks
    
    Returns:
        Response with request_id to track the search progress
    """
    
    # Validate input
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    logger.info(f"📥 New search request {request_id}: '{request.query}'")
    
    # Add background task
    background_tasks.add_task(
        process_search_request,
        request_id=request_id,
        query=request.query,
        include_patents=request.include_patents,
        include_papers=request.include_papers,
        top_k=request.top_k
    )
    
    # Return immediately with request ID
    return {
        "request_id": request_id,
        "message": "✅ Search request submitted. Use the request_id to check results.",
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/search/{request_id}", response_model=SearchResponse, tags=["Search"])
async def get_search_results(request_id: str):
    """
    Retrieve search results by request_id.
    
    Args:
        request_id: The unique identifier returned from /search endpoint
    
    Returns:
        SearchResponse with current status and results (if completed)
    """
    
    # Check cache first
    if request_id in search_cache:
        return search_cache[request_id]
    
    # Try to load from file
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            response = SearchResponse(**data)
            search_cache[request_id] = response
            return response
        except Exception as e:
            logger.error(f"Error loading results: {e}")
    
    # Not found
    raise HTTPException(
        status_code=404,
        detail=f"Request ID '{request_id}' not found. Please check the ID or submit a new search."
    )


@app.get("/search/status/{request_id}", tags=["Search"])
async def get_search_status(request_id: str):
    """
    Get the status of a search request without full results.
    
    Args:
        request_id: The unique identifier from /search endpoint
    
    Returns:
        Status information including progress
    """
    
    if request_id in search_cache:
        response = search_cache[request_id]
        return {
            "request_id": request_id,
            "status": response.status,
            "total_results": response.total_results,
            "timestamp": response.timestamp
        }
    
    filepath = os.path.join(RESULTS_DIR, f"{request_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "request_id": request_id,
                "status": data.get("status"),
                "total_results": data.get("total_results"),
                "timestamp": data.get("timestamp")
            }
        except Exception as e:
            logger.error(f"Error loading status: {e}")
    
    raise HTTPException(status_code=404, detail="Request ID not found")


@app.get("/requests", tags=["Search"])
async def list_all_requests():
    """
    List all search requests made in this session.
    
    Returns:
        List of request IDs and their status
    """
    
    requests_list = []
    
    # From cache
    for request_id, response in search_cache.items():
        requests_list.append({
            "request_id": request_id,
            "query": response.query,
            "status": response.status,
            "total_results": response.total_results,
            "timestamp": response.timestamp
        })
    
    # From files not in cache
    if os.path.exists(RESULTS_DIR):
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                request_id = filename.replace('.json', '')
                if request_id not in search_cache:
                    try:
                        with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                            data = json.load(f)
                        requests_list.append({
                            "request_id": request_id,
                            "query": data.get("query"),
                            "status": data.get("status"),
                            "total_results": data.get("total_results"),
                            "timestamp": data.get("timestamp")
                        })
                    except Exception as e:
                        logger.error(f"Error loading request {request_id}: {e}")
    
    return {
        "total_requests": len(requests_list),
        "requests": requests_list
    }


@app.get("/docs", tags=["Documentation"])
async def get_docs():
    """Interactive API documentation (Swagger UI)"""
    return {"message": "Visit /docs for interactive documentation"}


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("🚀 Starting Semantic Patent Search API...")
    logger.info(f"📁 Results directory: {os.path.abspath(RESULTS_DIR)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("🛑 Shutting down Semantic Patent Search API")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 Starting Uvicorn server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
