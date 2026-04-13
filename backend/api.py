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
from threading import Lock, Timer, Event
from queue import Queue
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    captcha_value: Optional[str] = None  # Optional if reusing session
    search_field: str = "Title"  # Dropdown field: Title, Abstract, Application Number, Complete Specification
    include_papers: bool = True
    top_k: Optional[int] = None
    ipr_limit: int = 25
    scholar_limit: int = 25
    email_id: Optional[str] = None


# ============================================================================
# GLOBAL STATE
# ============================================================================

search_cache = {}  # Stores SearchResponse objects for each request_id
RESULTS_DIR = "search_results"
patent_scraper = None
patent_scraper_lock = Lock()  # Prevents concurrent patent scraper access
cleanup_timers = {}
cleanup_lock = Lock()
CLEANUP_TTL_SECONDS = int(os.getenv("RESULT_CLEANUP_TTL_SECONDS", "300"))
CLEANUP_AFTER_EMAIL_SECONDS = int(os.getenv("RESULT_CLEANUP_AFTER_EMAIL_SECONDS", "60"))

# Request queue tracking for monitoring concurrent requests
request_queue = {}
queue_lock = Lock()

# Patent search queue system (allows multiple concurrent Selenium instances)
MAX_CONCURRENT_PATENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_PATENT_SEARCHES", "2"))
patent_search_queue = Queue()  # Thread-safe queue for patent search requests
active_patent_searches = 0  # Current count of active searches
patent_queue_lock = Lock()

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
        # Use the exact top_k value from API request
        candidate_k = min(len(normalized_docs), top_k)
        results = searcher.search(query, normalized_docs, top_k=candidate_k, similarity_threshold=0.30)
        logger.info(f"✅ BM25 ranking completed with top_k={top_k}")
        return results
        
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

    # Read from environment variables instead of hardcoding
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    FROM_EMAIL = os.getenv("FROM_EMAIL")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    if not SMTP_USER or not SMTP_PASS or not FROM_EMAIL:
        logger.warning("Email requested but SMTP credentials are not configured in .env; skipping email send.")
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
    msg["From"] = FROM_EMAIL
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
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
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


def track_active_request(request_id: str, status: str, query: str):
    """Track active/queued request for concurrent monitoring"""
    with queue_lock:
        request_queue[request_id] = {
            "query": query,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }


def untrack_request(request_id: str):
    """Remove request from active tracking"""
    with queue_lock:
        request_queue.pop(request_id, None)


def enqueue_patent_search(request_id: str, patent_request: PatentSearchRequest, background_tasks: BackgroundTasks):
    """Enqueue a patent search request to be processed"""
    global active_patent_searches
    
    patent_search_queue.put((request_id, patent_request))
    
    with patent_queue_lock:
        # If below max concurrent, start a worker
        if active_patent_searches < MAX_CONCURRENT_PATENT_SEARCHES:
            background_tasks.add_task(process_patent_search_worker)
            active_patent_searches += 1
            logger.info(f"Started patent search worker {active_patent_searches}/{MAX_CONCURRENT_PATENT_SEARCHES}")
        else:
            logger.info(f"Patent search queued. Workers at max ({MAX_CONCURRENT_PATENT_SEARCHES})")


def get_patent_queue_status():
    """Get current patent search queue status"""
    return {
        "active_searches": active_patent_searches,
        "max_concurrent": MAX_CONCURRENT_PATENT_SEARCHES,
        "queued_requests": patent_search_queue.qsize()
    }


async def process_patent_search_worker():
    """Worker that processes patent searches from queue"""
    global active_patent_searches
    
    while True:
        # Get next request from queue (non-blocking)
        if patent_search_queue.empty():
            with patent_queue_lock:
                active_patent_searches -= 1
            logger.info(f"Patent search worker idle. Active: {active_patent_searches}/{MAX_CONCURRENT_PATENT_SEARCHES}")
            break
        
        try:
            request_id, patent_request = patent_search_queue.get(block=False)
            logger.info(f"Processing patent search {request_id} from queue")
            
            # This will be replaced with actual patent search logic
            await execute_patent_search(request_id, patent_request)
            
        except Exception as e:
            logger.error(f"Error in patent search worker: {e}")
            with patent_queue_lock:
                active_patent_searches -= 1


async def execute_patent_search(request_id: str, request: PatentSearchRequest):
    """Execute the actual patent search (can run concurrently)"""
    try:
        global patent_scraper
        from backend.src.patent_extractor.patent_scraper import PatentScraperService

        track_active_request(request_id, "processing", f"Patent: {request.title}")
        
        # REUSE the existing patent_scraper from /patents/initiate if available
        # This preserves the CAPTCHA that was shown to the user
        if patent_scraper is not None:
            logger.info(f"Reusing existing patent scraper driver (preserves CAPTCHA from /patents/initiate)")
            patent_scraper_instance = patent_scraper
            reusing_driver = True
        else:
            # Fallback: Create a new instance if /patents/initiate wasn't called first
            logger.info(f"No existing patent scraper. Creating new instance.")
            patent_scraper_instance = PatentScraperService()
            if not patent_scraper_instance.init_driver():
                raise Exception("Failed to initialize WebDriver")
            reusing_driver = False
        
        patent_results = []
        try:
            # Perform search with the patent scraper instance
            patent_result = patent_scraper_instance.search_patents(
                request.title,
                request.captcha_value,
                search_field=request.search_field,
                max_results=request.ipr_limit
            )
            if patent_result.get("status") == "success":
                patent_results = patent_result.get("results", [])
            else:
                logger.warning(f"Patent scraping error: {patent_result.get('message')}")
        finally:
            # Only close if we created a new instance (don't close reused driver from /patents/initiate)
            if not reusing_driver:
                try:
                    patent_scraper_instance.close()
                except:
                    pass
            # NOTE: Do NOT reset patent_scraper here - keep browser session alive for multiple searches
            # This allows users to submit multiple searches with same CAPTCHA session

        # Run scholar scrape
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
        
        logger.info(f"✅ Patent search {request_id} completed (patents={len(normalized_patents)}, papers={len(normalized_scholar)})")
        track_active_request(request_id, "completed", f"Patent: {request.title}")
    
    except Exception as e:
        logger.error(f"❌ Error in patent search {request_id}: {e}")
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
        track_active_request(request_id, "failed", f"Patent: {request.title}")
    finally:
        untrack_request(request_id)


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
    Supports multiple concurrent requests.
    """
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1 or request.top_k > 400:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 400")
    
    request_id = str(uuid.uuid4())
    logger.info(f"📥 New search request {request_id}: '{request.query}' (user-supplied top_k={request.top_k})")

    # Track this request in queue
    track_active_request(request_id, "queued", request.query)
    
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
            track_active_request(request_id, "processing", request.query)
            
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
                track_active_request(request_id, "completed", request.query)
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
            
            logger.info(f"✅ Search request {request_id} completed with {len(results)} results")
            track_active_request(request_id, "completed", request.query)
        
        except Exception as e:
            logger.error(f"❌ Error processing search {request_id}: {e}")
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
            track_active_request(request_id, "failed", request.query)
        finally:
            untrack_request(request_id)
    
    # Add background task (runs immediately in background thread pool)
    background_tasks.add_task(process_search)
    
    return {
        "request_id": request_id,
        "message": "✅ Search request submitted. Processing in background.",
        "status": "queued",
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
    """List all search requests (cached and from disk)"""
    
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


@app.get("/requests/active", tags=["Search"])
async def get_active_requests():
    """Get currently active/queued requests (real-time concurrent requests)"""
    with queue_lock:
        active_requests = dict(request_queue)
    
    return {
        "active_count": len(active_requests),
        "active_requests": active_requests
    }


# ============================================================================
# PATENT SEARCH ENDPOINTS
# ============================================================================

@app.post("/patents/initiate", tags=["Patents"])
async def initiate_patent_search():
    """
    Initiate patent search by opening the patent office page.
    Returns CAPTCHA screenshot in base64 and available search fields.
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
            "search_fields": ["Title", "Abstract", "Application Number", "Complete Specification"],
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
    Submit patent search with title and optional CAPTCHA value.
    
    Args:
        title: Search query
        captcha_value: CAPTCHA value (optional if reusing session from /patents/initiate)
        search_field: Which field to search (Title, Abstract, etc.)
        
    Note: Patent searches use a queue system with multiple concurrent Selenium instances.
    Configurable via MAX_CONCURRENT_PATENT_SEARCHES environment variable (default: 2).
    
    To use without CAPTCHA (pipeline mode):
        1. Call /patents/initiate first (shows CAPTCHA and loads driver)
        2. Call /patents/search WITHOUT captcha_value (reuses same driver/session)
        3. Continue calling /patents/search for multiple searches with same session
    """
    
    if not request.title or len(request.title.strip()) == 0:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    # CAPTCHA is optional - only required if NOT reusing existing session
    if not request.captcha_value and patent_scraper is None:
        raise HTTPException(
            status_code=400, 
            detail="CAPTCHA value required. Call /patents/initiate first, or provide captcha_value to search without session"
        )
    
    if request.top_k and (request.top_k < 1 or request.top_k > 100):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    request_id = str(uuid.uuid4())
    logger.info(f"📥 Patent search request {request_id}: '{request.title}' (captcha_value={bool(request.captcha_value)})")

    # Track this request
    track_active_request(request_id, "queued", f"Patent: {request.title}")
    
    # Save an immediate processing placeholder
    search_cache[request_id] = SearchResponse(
        request_id=request_id,
        query=request.title,
        status="queued",
        total_results=0,
        timestamp=datetime.now().isoformat()
    )
    
    # Enqueue for processing
    enqueue_patent_search(request_id, request, background_tasks)
    
    queue_status = get_patent_queue_status()
    
    return {
        "request_id": request_id,
        "message": f"Patent search queued. Position in queue: {queue_status['queued_requests']}",
        "status": "queued",
        "queue_status": queue_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/patents/queue-status", tags=["Patents"])
async def get_patent_queue_update():
    """Get current patent search queue status"""
    return get_patent_queue_status()


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
