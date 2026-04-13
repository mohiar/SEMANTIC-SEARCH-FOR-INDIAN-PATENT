#!/usr/bin/env python3
"""
FastAPI Application for Semantic Search Pipeline
================================================
Handles user requests asynchronously with background task processing.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import logging
import uuid
from datetime import datetime
import smtplib
from email.message import EmailMessage
from threading import Lock, Timer
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (works even when launched from backend/).
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app (NOT Flask)
app = FastAPI(
    title="Semantic Patent Search API",
    description="Search for patents and academic papers using semantic search",
    version="1.0.0"
)

# Add CORS middleware
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


class SearchRequest(BaseModel):
    """User search request model"""
    query: str
    include_patents: bool = True
    include_papers: bool = True
    top_k: int = 10


class PatentSearchRequest(BaseModel):
    """Patent search request"""
    title: str
    captcha_value: str
    include_papers: bool = True
    top_k: Optional[int] = None
    ipr_limit: int = 25
    scholar_limit: int = 25
    email_id: Optional[str] = None


# ============================================================================
# GLOBAL STATE
# ============================================================================

search_cache = {}
RESULTS_DIR = "search_results"
patent_scraper = None
patent_scraper_lock = Lock()
cleanup_timers = {}
cleanup_lock = Lock()
CLEANUP_TTL_SECONDS = int(os.getenv("RESULT_CLEANUP_TTL_SECONDS", "300"))
CLEANUP_AFTER_EMAIL_SECONDS = int(os.getenv("RESULT_CLEANUP_AFTER_EMAIL_SECONDS", "60"))

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        if not combined_data:
            return []

        # Ensure missing abstracts do not zero-out documents in BM25.
        normalized_docs = []
        for doc in combined_data:
            abstract = (doc.get("abstract") or "").strip()
            title = (doc.get("title") or "").strip()
            if not abstract:
                abstract = title
            normalized = dict(doc)
            normalized["abstract"] = abstract
            normalized_docs.append(normalized)
        
        searcher = BM25SemanticSearch()
        # Widen candidate pool before final top-k cut so both sources can compete fairly.
        candidate_k = min(len(normalized_docs), max(top_k * 5, 50))
        results = searcher.search(query, normalized_docs, top_k=candidate_k, similarity_threshold=0.30)
        logger.info(f"✅ BM25 ranking completed")
        return results[:top_k]
        
    except ImportError:
        logger.warning("⚠️  BM25 module not found. Returning combined results without ranking.")
        return combined_data[:top_k]
    except Exception as e:
        logger.error(f"Error during BM25 ranking: {e}")
        return combined_data[:top_k]


def run_google_scholar_scrape(query: str, limit: int = 25) -> List[dict]:
    """Run Google Scholar scrape and return normalized paper list."""
    try:
        from backend.src.google_scholar.google_scholar_scraper import GoogleScholarScraper

        scraper = GoogleScholarScraper(db_name="scholar_data.db")
        papers = scraper.scrape_query_results(query, limit=limit)
        logger.info(f"✅ Google Scholar scrape completed with {len(papers)} papers")
        return papers
    except Exception as e:
        logger.error(f"❌ Google Scholar scraping failed: {e}")
        return []


def maybe_send_email_results(email_id: Optional[str], query: str, results: List[SearchResult], request_id: str):
    """Send result summary email when SMTP env is configured."""
    if not email_id:
        return False

    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = (os.getenv("SMTP_PASS"))
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT"))
    from_email = os.getenv("FROM_EMAIL")

    if not smtp_user or not smtp_pass or not from_email:
        logger.warning("Email requested but SMTP credentials are not configured; skipping email send.")
        return False

    body_lines = [
        f"Query: {query}",
        f"Request ID: {request_id}",
        f"Total Results: {len(results)}",
        "",
        "Top Results:"
    ]

    for idx, item in enumerate(results[:10], start=1):
        body_lines.append(f"{idx}. {item.title} [{item.source}]")
        if item.similarity_score is not None:
            body_lines.append(f"   Score: {item.similarity_score}")
        if item.url:
            body_lines.append(f"   URL: {item.url}")

    msg = EmailMessage()
    msg["Subject"] = f"Semantic Patent Search Results: {query}"
    msg["From"] = from_email
    msg["To"] = email_id
    msg.set_content("\n".join(body_lines))

    artifacts_dir = os.path.join(RESULTS_DIR, request_id)
    for artifact_name in ["combined_scraper_results.json", "semantic_results.json"]:
        artifact_path = os.path.join(artifacts_dir, artifact_name)
        if not os.path.exists(artifact_path):
            continue
        try:
            with open(artifact_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="application",
                    subtype="json",
                    filename=artifact_name
                )
        except Exception as attach_err:
            logger.warning(f"Could not attach {artifact_name}: {attach_err}")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"✅ Results email sent to {email_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send email to {email_id}: {e}")
        return False


def cleanup_request_data(request_id: str):
    """Remove cached and on-disk data for a request."""
    try:
        search_cache.pop(request_id, None)

        summary_file = os.path.join(RESULTS_DIR, f"{request_id}.json")
        if os.path.exists(summary_file):
            os.remove(summary_file)

        artifacts_dir = os.path.join(RESULTS_DIR, request_id)
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)

        logger.info(f"🧹 Cleaned up data for request {request_id}")
    except Exception as e:
        logger.error(f"Failed to clean up request {request_id}: {e}")
    finally:
        with cleanup_lock:
            cleanup_timers.pop(request_id, None)


def schedule_cleanup(request_id: str, delay_seconds: int):
    """Schedule delayed cleanup for request artifacts and cache."""
    with cleanup_lock:
        existing = cleanup_timers.get(request_id)
        if existing:
            existing.cancel()
        timer = Timer(delay_seconds, cleanup_request_data, args=[request_id])
        timer.daemon = True
        cleanup_timers[request_id] = timer
        timer.start()
    logger.info(f"⏳ Scheduled cleanup for request {request_id} in {delay_seconds}s")


def save_pipeline_artifacts(
    request_id: str,
    patent_results: List[dict],
    scholar_results: List[dict],
    combined_results: List[dict],
    semantic_results: List[SearchResult]
) -> dict:
    """Save per-request pipeline artifacts and zip archive."""
    artifacts_dir = os.path.join(RESULTS_DIR, request_id)
    os.makedirs(artifacts_dir, exist_ok=True)

    paths = {
        "patent": os.path.join(artifacts_dir, "patent_scraper_results.json"),
        "scholar": os.path.join(artifacts_dir, "google_scholar_results.json"),
        "combined": os.path.join(artifacts_dir, "combined_scraper_results.json"),
        "semantic": os.path.join(artifacts_dir, "semantic_results.json"),
    }

    with open(paths["patent"], "w", encoding="utf-8") as f:
        json.dump(patent_results, f, indent=2, ensure_ascii=False)
    with open(paths["scholar"], "w", encoding="utf-8") as f:
        json.dump(scholar_results, f, indent=2, ensure_ascii=False)
    with open(paths["combined"], "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    with open(paths["semantic"], "w", encoding="utf-8") as f:
        json.dump([item.dict() for item in semantic_results], f, indent=2, ensure_ascii=False)

    zip_base = os.path.join(artifacts_dir, "artifacts")
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=artifacts_dir)
    paths["zip"] = zip_path
    return paths


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "🚀 Semantic Patent Search API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }


# ============================================================================
# SEMANTIC SEARCH ENDPOINTS
# ============================================================================

@app.post("/search", response_model=dict, tags=["Search"])
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Submit a search query for processing.
    
    The search will be processed in the background.
    Use the returned request_id to check results.
    """
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.ipr_limit < 1 or request.ipr_limit > 200:
        raise HTTPException(status_code=400, detail="ipr_limit must be between 1 and 200")
    if request.scholar_limit < 1 or request.scholar_limit > 200:
        raise HTTPException(status_code=400, detail="scholar_limit must be between 1 and 200")
    if request.top_k is not None and (request.top_k < 1 or request.top_k > 400):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 400")
    
    request_id = str(uuid.uuid4())
    logger.info(f"📥 New search request {request_id}: '{request.query}'")

    # Save an immediate processing placeholder to prevent transient 404 on polling.
    search_cache[request_id] = SearchResponse(
        request_id=request_id,
        query=request.query,
        status="processing",
        total_results=0,
        timestamp=datetime.now().isoformat()
    )
    
    def process_search():
        """Background task to process search request"""
        try:
            response = SearchResponse(
                request_id=request_id,
                query=request.query,
                status="processing",
                total_results=0,
                timestamp=datetime.now().isoformat()
            )
            
            # Load data from sources
            google_scholar_data = []
            patent_data = []
            
            if request.include_papers:
                google_scholar_data = load_json_data("google_scholar_results.json")
                logger.info(f"✅ Loaded {len(google_scholar_data)} Google Scholar papers")
            
            if request.include_patents:
                patent_data = load_json_data("patent_search_results.json")
                logger.info(f"✅ Loaded {len(patent_data)} patents")
            
            if not google_scholar_data and not patent_data:
                response.status = "failed"
                response.error_message = "No data available. Please run the scrapers first."
                search_cache[request_id] = response
                save_search_results(request_id, response)
                return
            
            # Combine and rank
            combined_data = combine_sources(google_scholar_data, patent_data)
            ranked_results = apply_bm25_ranking(combined_data, request.query, request.top_k)
            
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
            
            search_cache[request_id] = response
            save_search_results(request_id, response)
            
            logger.info(f"✅ Search request {request_id} completed")
        
        except Exception as e:
            logger.error(f"❌ Error processing search: {e}")
            response = SearchResponse(
                request_id=request_id,
                query=request.query,
                status="failed",
                total_results=0,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            search_cache[request_id] = response
            save_search_results(request_id, response)
    
    # Add background task
    background_tasks.add_task(process_search)
    
    return {
        "request_id": request_id,
        "message": "✅ Search request submitted.",
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/search/{request_id}", response_model=SearchResponse, tags=["Search"])
async def get_search_results(request_id: str):
    """Retrieve search results by request_id"""
    
    if request_id in search_cache:
        return search_cache[request_id]
    
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
    
    raise HTTPException(
        status_code=404,
        detail=f"Request ID '{request_id}' not found."
    )


@app.get("/search/status/{request_id}", tags=["Search"])
async def get_search_status(request_id: str):
    """Get the status of a search request"""
    
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


@app.get("/search/{request_id}/download/{artifact}", tags=["Search"])
async def download_search_artifact(request_id: str, artifact: str):
    """
    Download per-request artifact files.
    Supported artifacts: combined, semantic, patent, scholar, zip
    """
    artifact_map = {
        "combined": "combined_scraper_results.json",
        "semantic": "semantic_results.json",
        "patent": "patent_scraper_results.json",
        "scholar": "google_scholar_results.json",
        "zip": "artifacts.zip",
    }
    if artifact not in artifact_map:
        raise HTTPException(status_code=400, detail="Invalid artifact type")

    file_path = os.path.join(RESULTS_DIR, request_id, artifact_map[artifact])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = "application/zip" if artifact == "zip" else "application/json"
    return FileResponse(file_path, media_type=media_type, filename=artifact_map[artifact])


@app.get("/requests", tags=["Search"])
async def list_all_requests():
    """List all search requests"""
    
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
    
    # From files
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


# ============================================================================
# PATENT SEARCH ENDPOINTS
# ============================================================================

@app.post("/patents/initiate", tags=["Patents"])
async def initiate_patent_search():
    """
    Initiate patent search by opening the patent office page.
    Returns CAPTCHA screenshot in base64.
    """
    logger.info("Initiating patent search...")
    
    lock_acquired = False
    try:
        global patent_scraper
        lock_acquired = patent_scraper_lock.acquire(blocking=False)
        if not lock_acquired:
            raise HTTPException(
                status_code=409,
                detail="Patent scraper is busy with another request. Please try again shortly."
            )
        
        # Import here to avoid circular imports
        from backend.src.patent_extractor.patent_scraper import PatentScraperService
        
        # Initialize scraper if not already done
        if patent_scraper is None:
            patent_scraper = PatentScraperService()
            if not patent_scraper.init_driver():
                raise Exception("Failed to initialize WebDriver")
        
        # Get CAPTCHA screenshot
        captcha_image = patent_scraper.get_captcha_screenshot()
        
        if not captcha_image:
            raise Exception("Failed to capture CAPTCHA")
        
        return {
            "status": "ready",
            "message": "Patent search page loaded. Please solve the CAPTCHA.",
            "captcha_image": f"data:image/png;base64,{captcha_image}",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating patent search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if lock_acquired:
            patent_scraper_lock.release()


@app.post("/patents/search", response_model=dict, tags=["Patents"])
async def search_patents(request: PatentSearchRequest, background_tasks: BackgroundTasks):
    """
    Submit patent search with title and CAPTCHA value.
    """
    
    if not request.title or len(request.title.strip()) == 0:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    if not request.captcha_value or len(request.captcha_value.strip()) == 0:
        raise HTTPException(status_code=400, detail="CAPTCHA value cannot be empty")
    
    if request.top_k < 1 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    if patent_scraper_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Patent scraper is busy with another request. Please try again shortly."
        )
    
    request_id = str(uuid.uuid4())
    logger.info(f"Patent search request {request_id}: '{request.title}'")

    # Save an immediate processing placeholder to prevent transient 404 on polling.
    search_cache[request_id] = SearchResponse(
        request_id=request_id,
        query=request.title,
        status="processing",
        total_results=0,
        timestamp=datetime.now().isoformat()
    )
    
    def process_patent_search():
        """Background task: patent scrape -> scholar scrape -> BM25 ranking"""
        try:
            global patent_scraper
            from backend.src.patent_extractor.patent_scraper import PatentScraperService

            patent_results = []
            with patent_scraper_lock:
                if patent_scraper is None:
                    patent_scraper = PatentScraperService()
                    patent_scraper.init_driver()
                
                # Perform search
                patent_result = patent_scraper.search_patents(
                    request.title,
                    request.captcha_value,
                    max_results=request.ipr_limit
                )
                if patent_result.get("status") == "success":
                    patent_results = patent_result.get("results", [])
                else:
                    logger.warning(f"Patent scraping error: {patent_result.get('message')}")

            # Run scholar scrape outside patent lock
            scholar_results = []
            if request.include_papers:
                scholar_results = run_google_scholar_scrape(request.title, limit=request.scholar_limit)

            # Normalize and combine
            normalized_patents = []
            for patent in patent_results:
                normalized_patents.append({
                    "title": patent.get("title", ""),
                    "abstract": patent.get("abstract", ""),
                    "url": patent.get("patent_url", ""),
                    "application_number": patent.get("application_number", ""),
                    "source": "Indian Patent Office",
                })

            normalized_scholar = []
            for paper in scholar_results:
                normalized_scholar.append({
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "") or paper.get("snippet", ""),
                    "url": paper.get("paper_url", ""),
                    "authors": paper.get("authors", ""),
                    "source": "Google Scholar",
                })

            combined_data = normalized_patents + normalized_scholar
            final_top_k = request.top_k or min(len(combined_data), request.ipr_limit + request.scholar_limit)
            ranked_results = apply_bm25_ranking(combined_data, request.title, final_top_k)
            
            # Create response
            response = SearchResponse(
                request_id=request_id,
                query=request.title,
                status="completed",
                total_results=0,
                timestamp=datetime.now().isoformat(),
                error_message=None
            )
            
            for item in ranked_results:
                response.results.append(SearchResult(
                    title=item.get("title", ""),
                    abstract=item.get("abstract", ""),
                    source=item.get("source", ""),
                    url=item.get("url"),
                    application_number=item.get("application_number"),
                    authors=item.get("authors"),
                    similarity_score=item.get("similarity_score")
                ))

            response.total_results = len(response.results)
            if not combined_data:
                response.status = "failed"
                response.error_message = "No data found from patent and scholar scraping."

            artifact_paths = save_pipeline_artifacts(
                request_id=request_id,
                patent_results=normalized_patents,
                scholar_results=normalized_scholar,
                combined_results=combined_data,
                semantic_results=response.results
            )
            
            search_cache[request_id] = response
            save_search_results(request_id, response)
            email_sent = maybe_send_email_results(request.email_id, request.title, response.results, request_id)
            cleanup_delay = CLEANUP_AFTER_EMAIL_SECONDS if email_sent else CLEANUP_TTL_SECONDS
            schedule_cleanup(request_id, cleanup_delay)
            
            logger.info(
                f"Combined search {request_id} completed "
                f"(patents={len(normalized_patents)}, papers={len(normalized_scholar)}, final={response.total_results}, "
                f"artifacts={artifact_paths.get('zip')})"
            )
        
        except Exception as e:
            logger.error(f"Error processing patent search: {e}")
            response = SearchResponse(
                request_id=request_id,
                query=request.title,
                status="failed",
                total_results=0,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            search_cache[request_id] = response
            save_search_results(request_id, response)
            schedule_cleanup(request_id, CLEANUP_TTL_SECONDS)
    
    background_tasks.add_task(process_patent_search)
    
    return {
        "request_id": request_id,
        "message": "Patent search submitted.",
        "status": "processing",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    logger.info("🚀 Starting Semantic Patent Search API...")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global patent_scraper
    if patent_scraper:
        patent_scraper.close()
    logger.info("🛑 Shutting down API")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
