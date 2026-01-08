#!/usr/bin/env python3
"""
Google Scholar End-to-End Scraper
==================================
This script takes a user query, searches Google Scholar, clicks all links on page 1,
extracts data from each paper, and stores the results in an SQLite database.

Author: AI Assistant
Date: October 2024
"""

import sqlite3
import time
import logging
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleScholarScraper:
    def __init__(self, db_name="scholar_data.db"):
        """Initialize the scraper with database setup."""
        self.db_name = db_name
        self.driver = None
        self.setup_database()
        self.setup_requests_session()

    def setup_database(self):
        """Create SQLite database and tables."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Create main papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT NOT NULL,
                    title TEXT,
                    authors TEXT,
                    publication_info TEXT,
                    snippet TEXT,
                    paper_url TEXT UNIQUE,
                    citations_count TEXT,
                    pdf_url TEXT,
                    full_text TEXT,
                    abstract TEXT,
                    keywords TEXT,
                    publication_year INTEGER,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create search history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    papers_found INTEGER,
                    papers_processed INTEGER,
                    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
            logger.info(f"Database {self.db_name} initialized successfully")

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def setup_requests_session(self):
        """Setup requests session with retry strategy."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def init_selenium_driver(self):
        """Initialize Selenium WebDriver with anti-detection options."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Comment this for debugging
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("Chrome WebDriver initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            return False

    def search_google_scholar(self, query):
        """Search Google Scholar and return search results page."""
        try:
            base_url = "https://scholar.google.com/scholar"
            search_url = f"{base_url}?q={query.replace(' ', '+')}"

            self.driver.get(search_url)
            time.sleep(3)  # Wait for page to load

            # Wait for search results to appear
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gs_ri"))
            )

            logger.info(f"Successfully loaded search results for: {query}")
            return True

        except TimeoutException:
            logger.error("Timeout waiting for search results")
            return False
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return False

    def extract_search_results(self):
        """Extract all paper links and metadata from search results page."""
        papers = []
        try:
            # Find all paper result containers
            result_containers = self.driver.find_elements(By.CLASS_NAME, "gs_ri")
            logger.info(f"Found {len(result_containers)} papers on search results page")

            for i, container in enumerate(result_containers):
                try:
                    paper_data = {}

                    # Extract title and link
                    title_element = container.find_element(By.CLASS_NAME, "gs_rt")
                    paper_data['title'] = title_element.text.strip()

                    # Try to get the paper URL from the title link
                    try:
                        title_link = title_element.find_element(By.TAG_NAME, "a")
                        paper_data['paper_url'] = title_link.get_attribute("href")
                    except NoSuchElementException:
                        paper_data['paper_url'] = None

                    # Extract authors and publication info
                    try:
                        authors_element = container.find_element(By.CLASS_NAME, "gs_a")
                        paper_data['authors'] = authors_element.text.strip()
                    except NoSuchElementException:
                        paper_data['authors'] = None

                    # Extract snippet/description
                    try:
                        snippet_element = container.find_element(By.CLASS_NAME, "gs_rs")
                        paper_data['snippet'] = snippet_element.text.strip()
                    except NoSuchElementException:
                        paper_data['snippet'] = None

                    # Extract citation count
                    try:
                        citation_element = container.find_element(By.XPATH, ".//a[contains(@href, 'cites=')]")
                        citation_text = citation_element.text.strip()
                        paper_data['citations_count'] = citation_text
                    except NoSuchElementException:
                        paper_data['citations_count'] = None

                    # Look for PDF links
                    try:
                        pdf_element = container.find_element(By.XPATH, ".//a[contains(@href, '.pdf') or contains(text(), '[PDF]')]")
                        paper_data['pdf_url'] = pdf_element.get_attribute("href")
                    except NoSuchElementException:
                        paper_data['pdf_url'] = None

                    papers.append(paper_data)
                    logger.info(f"Extracted metadata for paper {i+1}: {paper_data['title'][:50]}...")

                except Exception as e:
                    logger.warning(f"Error extracting data for paper {i+1}: {e}")
                    continue

            return papers

        except Exception as e:
            logger.error(f"Error extracting search results: {e}")
            return []

    def extract_full_paper_content(self, paper_url, timeout=20):
        """Extract full content from individual paper pages using Selenium to handle JS."""
        if not paper_url:
            return None

        try:
            # Use the Selenium driver to get the page, allowing JS to render
            self.driver.get(paper_url)
            # Wait for a potential abstract element to be loaded
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="abstract"], section[class*="abstract"], .abstract, #abstract, .article-details__section.abstract'))
            )
            
            # Give it a little more time for content to settle
            time.sleep(2)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            content_data = {
                'full_text': '',
                'abstract': '',
                'keywords': '',
                'publication_year': None
            }

            # More specific selectors can be added here for sites like IEEE
            # Example for some IEEE pages: '.article-details__section.abstract .u-mb-1'
            abstract_selectors = [
                'div.abstract-text', # For Springer
                '.article-details__section.abstract', # For some IEEE pages
                'div[class*="abstract"]',
                'section[class*="abstract"]',
                'p[class*="abstract"]',
                '.abstract',
                '#abstract'
            ]

            for selector in abstract_selectors:
                abstract_elem = soup.select_one(selector)
                if abstract_elem:
                    content_data['abstract'] = abstract_elem.get_text(separator=' ', strip=True)
                    logger.info(f"Successfully extracted abstract from {paper_url}")
                    break
            
            if not content_data['abstract']:
                 logger.warning(f"Could not find abstract on {paper_url} with current selectors.")


            # Look for keywords
            keyword_selectors = [
                'div[class*="keyword"]',
                'span[class*="keyword"]',
                '.keywords',
                '#keywords'
            ]

            for selector in keyword_selectors:
                keywords_elem = soup.select_one(selector)
                if keywords_elem:
                    content_data['keywords'] = keywords_elem.get_text().strip()
                    break

            # Extract full text (first few paragraphs)
            paragraphs = soup.find_all('p')[:10]  # Limit to first 10 paragraphs
            full_text = ' '.join([p.get_text().strip() for p in paragraphs])
            content_data['full_text'] = full_text[:2000]  # Limit text length

            # Try to extract publication year
            import re
            text_content = soup.get_text()
            year_match = re.search(r'\b(19|20)\d{2}\b', text_content)
            if year_match:
                content_data['publication_year'] = int(year_match.group())

            return content_data

        except TimeoutException:
            logger.warning(f"Timeout waiting for abstract content on {paper_url}. Page might be slow or content not found.")
            return None
        except Exception as e:
            logger.warning(f"Error extracting content from {paper_url}: {e}")
            return None

    def save_papers_to_db(self, papers, query):
        """Save papers to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            saved_count = 0
            for paper in papers:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO papers 
                        (search_query, title, authors, publication_info, snippet, 
                         paper_url, citations_count, pdf_url, full_text, abstract, 
                         keywords, publication_year)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        query,
                        paper.get('title'),
                        paper.get('authors'),
                        paper.get('publication_info'),
                        paper.get('snippet'),
                        paper.get('paper_url'),
                        paper.get('citations_count'),
                        paper.get('pdf_url'),
                        paper.get('full_text'),
                        paper.get('abstract'),
                        paper.get('keywords'),
                        paper.get('publication_year')
                    ))

                    if cursor.rowcount > 0:
                        saved_count += 1

                except Exception as e:
                    logger.warning(f"Error saving paper to database: {e}")
                    continue

            # Save search history
            cursor.execute("""
                INSERT INTO search_history (query, papers_found, papers_processed)
                VALUES (?, ?, ?)
            """, (query, len(papers), saved_count))

            conn.commit()
            conn.close()

            logger.info(f"Successfully saved {saved_count} papers to database")
            return saved_count

        except Exception as e:
            logger.error(f"Database save error: {e}")
            return 0

    def scrape_scholar_query(self, query):
        """Main method to scrape Google Scholar for a query."""
        logger.info(f"Starting scrape for query: {query}")

        # Initialize WebDriver
        if not self.init_selenium_driver():
            return False

        try:
            # Search Google Scholar
            if not self.search_google_scholar(query):
                return False

            # --- HANDLE COOKIE BANNERS ---
            try:
                # Wait for a moment for any cookie banners to appear
                time.sleep(2)
                # Find and click the "Accept all" button. This selector is generic.
                accept_button = self.driver.find_element(By.XPATH, "//button[contains(., 'Accept all')] | //button[contains(., 'I agree')]")
                accept_button.click()
                logger.info("Clicked the cookie consent button.")
                time.sleep(2) # Wait for the banner to disappear
            except NoSuchElementException:
                logger.info("No cookie consent banner found, continuing...")
            except Exception as e:
                logger.warning(f"Could not click cookie button, it might not be an issue: {e}")
            # -----------------------------

            # This will hold all papers from all pages
            all_papers = []
            target_papers = 50
            page_count = 1

            while len(all_papers) < target_papers:
                logger.info(f"Scraping page {page_count}...")
                
                # Extract search results from the current page
                papers_on_page = self.extract_search_results()
                if not papers_on_page:
                    logger.warning("No more papers found on this page. Stopping pagination.")
                    break
                
                all_papers.extend(papers_on_page)
                logger.info(f"Collected {len(all_papers)} papers so far.")

                # Find and click the 'Next' button
                try:
                    # A more specific selector for the "Next" link at the bottom
                    next_button = self.driver.find_element(By.CSS_SELECTOR, '#gs_n a:last-child')
                    
                    # Scroll to the button to make sure it's clickable
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1) # Wait a moment after scrolling

                    # Final check if it's the "Next" link
                    if "Next" not in next_button.text:
                         logger.info("Last link is not a 'Next' button. Reached the end.")
                         break

                    # Use a JavaScript click to bypass potential overlays
                    self.driver.execute_script("arguments[0].click();", next_button)
                    
                    page_count += 1
                    logger.info("Navigating to the next page.")
                    
                    # Wait for the next page's results to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "gs_ri"))
                    )
                    time.sleep(3) # Give page time to settle

                except NoSuchElementException:
                    logger.info("No 'Next' button found. Reached the last page.")
                    break
            
            papers = all_papers[:target_papers] # Trim to exactly the target number if we overshot
            logger.info(f"Finished collecting paper metadata. Total found: {len(papers)}")

            if not papers:
                logger.warning("No papers found in search results")
                return False

            # --- MODIFICATION: Use snippet as abstract and skip individual page scraping ---
            logger.info("Processing collected papers...")
            processed_papers = []
            for paper in papers:
                # Use the snippet from the search results page as the abstract
                if paper.get('snippet'):
                    paper['abstract'] = paper['snippet']
                processed_papers.append(paper)
            
            # Save to database
            saved_count = self.save_papers_to_db(processed_papers, query)
            logger.info(f"Scraping completed. Saved {saved_count} papers to database.")

            return True

        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return False

        finally:
            if self.driver:
                self.driver.quit()

    def display_results(self, query=None):
        """Display results from database."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            if query:
                cursor.execute('SELECT * FROM papers WHERE search_query = ? ORDER BY id DESC', (query,))
            else:
                cursor.execute('SELECT * FROM papers ORDER BY id DESC LIMIT 20')

            results = cursor.fetchall()
            conn.close()

            if not results:
                print("No results found in database.")
                return

            print(f"\n=== Database Results ({'for query: ' + query if query else 'Recent 20'}) ===")
            for row in results:
                print(f"\nID: {row[0]}")
                print(f"Title: {row[2]}")
                print(f"Authors: {row[3]}")
                print(f"URL: {row[6]}")
                print(f"Citations: {row[7]}")
                print(f"Snippet: {row[4][:100]}..." if row[4] else "No snippet")
                print("-" * 80)

        except Exception as e:
            logger.error(f"Error displaying results: {e}")

def main():
    """Main function to run the scraper."""
    print("Google Scholar End-to-End Scraper")
    print("=" * 40)

    # Get user input
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid search query.")
            continue

        # Initialize scraper
        scraper = GoogleScholarScraper()

        try:
            # Perform scraping
            success = scraper.scrape_scholar_query(query)

            if success:
                print("\nScraping completed successfully!")

                # Ask if user wants to see results
                show_results = input("\nWould you like to see the results? (y/n): ").strip().lower()
                if show_results in ['y', 'yes']:
                    scraper.display_results(query)
            else:
                print("\nScraping failed. Check the logs for details.")

        except KeyboardInterrupt:
            print("\nScraping interrupted by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
