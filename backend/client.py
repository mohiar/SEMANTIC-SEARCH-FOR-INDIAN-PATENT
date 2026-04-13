#!/usr/bin/env python3
"""
Client script to interact with the Semantic Patent Search API
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load root .env
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

BASE_URL = os.getenv("NEXT_PUBLIC_API_BASE_URL", "http://localhost:8000")


def submit_search(query: str, include_patents: bool = True, 
                 include_papers: bool = True, top_k: int = 10) -> Optional[str]:
    """Submit a search request and return the request_id"""
    print(f"\n📤 Submitting search request: '{query}'")
    
    payload = {
        "query": query,
        "include_patents": include_patents,
        "include_papers": include_papers,
        "top_k": top_k
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search", json=payload)
        response.raise_for_status()
        
        data = response.json()
        request_id = data.get("request_id")
        
        print(f"✅ Request submitted!")
        print(f"📌 Request ID: {request_id}")
        print(f"💬 Message: {data.get('message')}")
        
        return request_id
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error submitting request: {e}")
        return None


def check_status(request_id: str) -> dict:
    """Check the status of a search request"""
    try:
        response = requests.get(f"{BASE_URL}/search/status/{request_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking status: {e}")
        return None


def get_results(request_id: str) -> Optional[dict]:
    """Retrieve the results of a completed search"""
    try:
        response = requests.get(f"{BASE_URL}/search/{request_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error retrieving results: {e}")
        return None


def wait_for_results(request_id: str, timeout: int = 300, poll_interval: int = 5):
    """Wait for search results to be ready"""
    print(f"\n⏳ Waiting for results... (timeout: {timeout}s)")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = check_status(request_id)
        
        if status:
            print(f"⏱️  Status: {status.get('status')} | Results: {status.get('total_results')}")
            
            if status.get('status') == 'completed':
                print("✅ Results are ready!")
                return True
            elif status.get('status') == 'failed':
                print(f"❌ Search failed: {status}")
                return False
        
        time.sleep(poll_interval)
    
    print(f"⏰ Timeout reached after {timeout}s")
    return False


def display_results(results: dict):
    """Display search results in a formatted way"""
    print("\n" + "="*70)
    print(f"🔍 SEARCH RESULTS FOR: '{results.get('query')}'")
    print("="*70)
    
    print(f"Status: {results.get('status')}")
    print(f"Total Results: {results.get('total_results')}")
    print(f"Timestamp: {results.get('timestamp')}")
    
    if results.get('error_message'):
        print(f"❌ Error: {results.get('error_message')}")
        return
    
    print("\n" + "-"*70)
    print("TOP RESULTS:")
    print("-"*70)
    
    for i, result in enumerate(results.get('results', []), 1):
        print(f"\n{i}. {result.get('title', 'N/A')}")
        print(f"   📌 Source: {result.get('source')}")
        if result.get('application_number'):
            print(f"   🔢 Patent Number: {result.get('application_number')}")
        if result.get('authors'):
            print(f"   👥 Authors: {result.get('authors')}")
        print(f"   📄 Abstract: {result.get('abstract', 'N/A')[:150]}...")
        if result.get('url'):
            print(f"   🔗 URL: {result.get('url')}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Interactive CLI for the API"""
    print("\n" + "="*70)
    print("🔍 SEMANTIC PATENT SEARCH CLIENT")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("1. Submit new search")
        print("2. Check status of existing search")
        print("3. Get results of completed search")
        print("4. Submit search and wait for results")
        print("5. List all requests")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                request_id = submit_search(query)
        
        elif choice == "2":
            request_id = input("Enter request ID: ").strip()
            if request_id:
                status = check_status(request_id)
                if status:
                    print(json.dumps(status, indent=2))
        
        elif choice == "3":
            request_id = input("Enter request ID: ").strip()
            if request_id:
                results = get_results(request_id)
                if results:
                    display_results(results)
        
        elif choice == "4":
            query = input("Enter search query: ").strip()
            if query:
                request_id = submit_search(query)
                if request_id:
                    if wait_for_results(request_id):
                        results = get_results(request_id)
                        if results:
                            display_results(results)
        
        elif choice == "5":
            try:
                response = requests.get(f"{BASE_URL}/requests")
                response.raise_for_status()
                requests_data = response.json()
                print(f"\nTotal Requests: {requests_data.get('total_requests')}")
                for req in requests_data.get('requests', [])[:10]:
                    print(f"  - {req.get('request_id')}: {req.get('query')} ({req.get('status')})")
            except requests.exceptions.RequestException as e:
                print(f"❌ Error: {e}")
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
