import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
from urllib.parse import quote
import sys

class GoogleScholarScraper:
    def __init__(self, db_name='google_scholar_data.db'):
        """Initialize the scraper with database connection"""
        self.db_name = db_name
        self.setup_database()
        self.session = requests.Session()
        # Add headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def setup_database(self):
        """Create database and table if they don't exist - FIXED VERSION"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            # Drop existing table if it has wrong schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scholar_articles';")
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(scholar_articles);")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # If search_query column doesn't exist, recreate table
                if 'search_query' not in column_names:
                    print("Updating database schema...")
                    cursor.execute('DROP TABLE scholar_articles')
            
            # Create table with correct schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scholar_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT,
                    title TEXT,
                    authors TEXT,
                    publication_info TEXT,
                    url TEXT,
                    snippet TEXT,
                    citations_count TEXT,
                    year TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            conn.commit()
            conn.close()
            print("✓ Database setup complete")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            sys.exit(1)
    
    def get_scholar_page(self, query, start=0):
        """Fetch Google Scholar search results page"""
        base_url = "https://scholar.google.com/scholar"
        
        # Construct URL manually to avoid encoding issues
        url = f"{base_url}?q={quote(query)}&hl=en&as_sdt=0,5&start={start}"
        
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
    
    def parse_article_data(self, article_soup):
        """Extract data from a single article element"""
        data = {
            'title': '',
            'authors': '',
            'publication_info': '',
            'url': '',
            'snippet': '',
            'citations_count': '',
            'year': ''
        }
        
        try:
            # Extract title and URL
            title_elem = article_soup.find('h3', class_='gs_rt')
            if title_elem:
                title_link = title_elem.find('a')
                if title_link:
                    data['title'] = title_link.get_text(strip=True)
                    data['url'] = title_link.get('href', '')
                else:
                    data['title'] = title_elem.get_text(strip=True)
            
            # Extract authors and publication info
            authors_elem = article_soup.find('div', class_='gs_a')
            if authors_elem:
                authors_text = authors_elem.get_text(strip=True)
                data['authors'] = authors_text
                data['publication_info'] = authors_text  # Store full publication info
                # Try to extract year from publication info
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
                if year_match:
                    data['year'] = year_match.group()
            
            # Extract snippet
            snippet_elem = article_soup.find('div', class_='gs_rs')
            if snippet_elem:
                data['snippet'] = snippet_elem.get_text(strip=True)
            
            # Extract citation count
            citation_elem = article_soup.find('div', class_='gs_fl')
            if citation_elem:
                citation_link = citation_elem.find('a', string=lambda text: text and 'Cited by' in text)
                if citation_link:
                    data['citations_count'] = citation_link.get_text(strip=True)
            
        except Exception as e:
            print(f"Error parsing article: {e}")
        
        return data
    
    def scrape_results(self, query, max_pages=1):
        """Scrape Google Scholar results for a given query"""
        all_articles = []
        
        for page in range(max_pages):
            start = page * 10
            print(f"Scraping page {page + 1}...")
            
            html = self.get_scholar_page(query, start)
            if not html:
                print(f"Failed to fetch page {page + 1}")
                continue
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all article containers
            articles = soup.find_all('div', class_='gs_ri')
            
            if not articles:
                print(f"No articles found on page {page + 1}")
                break
            
            for article in articles:
                article_data = self.parse_article_data(article)
                article_data['search_query'] = query  # Add search query to each article
                all_articles.append(article_data)
            
            print(f"Found {len(articles)} articles on page {page + 1}")
        
        return all_articles
    
    def save_to_database(self, articles):
        """Save scraped articles to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            for article in articles:
                cursor.execute('''
                    INSERT INTO scholar_articles 
                    (search_query, title, authors, publication_info, url, snippet, citations_count, year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['search_query'],
                    article['title'],
                    article['authors'],
                    article['publication_info'],
                    article['url'],
                    article['snippet'],
                    article['citations_count'],
                    article['year']
                ))
            
            conn.commit()
            conn.close()
            print(f"✓ Saved {len(articles)} articles to database")
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
    
    def display_results(self, articles):
        """Display scraped results in a formatted way"""
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS: {len(articles)} articles found")
        print(f"{'='*80}")
        
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Authors: {article['authors']}")
            if article['url']:
                print(f"   URL: {article['url']}")
            if article['snippet']:
                print(f"   Snippet: {article['snippet'][:200]}...")
            if article['citations_count']:
                print(f"   Citations: {article['citations_count']}")
            print(f"   {'-'*70}")
    
    def search_database(self, search_term=None):
        """Search existing data in the database"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            if search_term:
                cursor.execute('''
                    SELECT * FROM scholar_articles 
                    WHERE title LIKE ? OR authors LIKE ? OR search_query LIKE ?
                    ORDER BY scraped_at DESC
                ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
            else:
                cursor.execute('SELECT * FROM scholar_articles ORDER BY scraped_at DESC LIMIT 50')
            
            results = cursor.fetchall()
            conn.close()
            
            return results
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []

def main():
    """Main function to run the scraper"""
    scraper = GoogleScholarScraper()
    
    print("Google Scholar Scraper")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Scrape new articles")
        print("2. Search existing database")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Get search query from user
            query = input("\nEnter search title/keyword: ").strip()
            if not query:
                print("Please enter a valid search term.")
                continue
            
            # Get number of pages to scrape
            try:
                max_pages = int(input("Number of pages to scrape (default 1): ") or "1")
                max_pages = max(1, min(max_pages, 10))  # Limit to 10 pages max
            except ValueError:
                max_pages = 1
            
            print(f"\nScraping Google Scholar for: '{query}'")
            print(f"Pages to scrape: {max_pages}")
            
            # Scrape results
            articles = scraper.scrape_results(query, max_pages)
            
            if articles:
                # Display results
                scraper.display_results(articles)
                
                # Save to database
                scraper.save_to_database(articles)
                
                print(f"\n✓ Successfully scraped and saved {len(articles)} articles!")
            else:
                print("No articles found for the given query.")
        
        elif choice == '2':
            search_term = input("\nEnter search term (or press Enter for recent articles): ").strip()
            results = scraper.search_database(search_term if search_term else None)
            
            if results:
                print(f"\nFound {len(results)} articles in database:")
                print("-" * 80)
                for result in results[:10]:  # Show first 10 results
                    print(f"Title: {result[2]}")
                    print(f"Authors: {result[3]}")
                    print(f"Search Query: {result[1]}")
                    print(f"Scraped: {result[9]}")
                    print("-" * 80)
            else:
                print("No articles found in database.")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
