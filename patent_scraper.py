#!/usr/bin/env python3
"""
Google Scholar Patent Scraper
=============================
This script takes a user query, searches Google Scholar for patents,
and scrapes the title and abstract snippet for a specified number of results.
It handles pagination automatically and saves the output to a JSON file
named after the search query.

Author: AI Assistant
Date: November 2025
"""

import json
import time
import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_selenium_driver():
    """Initialize Selenium WebDriver with anti-detection options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.info("Chrome WebDriver initialized successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver: {e}")
        return None

def scrape_google_patents(query, num_patents=50):
    """
    Scrapes Google Scholar for patents based on a query.

    Args:
        query (str): The search term for patents.
        num_patents (int): The target number of patents to scrape.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              title and abstract of a patent.
    """
    driver = init_selenium_driver()
    if not driver:
        return []

    patents = []
    try:
        # Construct the URL to search for patents specifically (&tbs=ptk)
        search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&tbs=ptk"
        driver.get(search_url)
        logger.info(f"Navigated to search results for query: '{query}'")

        # --- Optional: Handle Cookie Banners ---
        try:
            time.sleep(2)
            accept_button = driver.find_element(By.XPATH, "//button[contains(., 'Accept all')]")
            accept_button.click()
            logger.info("Clicked the cookie consent button.")
            time.sleep(2)
        except NoSuchElementException:
            logger.info("No cookie consent banner found, continuing...")
        except Exception as e:
            logger.warning(f"Could not click cookie button (this may not be an issue): {e}")
        # ------------------------------------

        page_count = 1
        while len(patents) < num_patents:
            logger.info(f"Scraping page {page_count}...")
            
            # Wait for results to be present on the page
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gs_ri"))
            )

            # Extract patent data from the current page
            result_containers = driver.find_elements(By.CLASS_NAME, "gs_ri")
            if not result_containers:
                logger.warning("No more results found on this page. Stopping.")
                break

            for container in result_containers:
                try:
                    title_element = container.find_element(By.CLASS_NAME, "gs_rt").find_element(By.TAG_NAME, "a")
                    title = title_element.text.strip()
                    
                    snippet_element = container.find_element(By.CLASS_NAME, "gs_rs")
                    abstract_snippet = snippet_element.text.strip()

                    patents.append({
                        "title": title,
                        "abstract": abstract_snippet
                    })
                except NoSuchElementException:
                    continue # Skip if a container is missing title or snippet
            
            logger.info(f"Collected {len(patents)} patents so far.")

            # Check if we have enough or if there's a next page
            if len(patents) >= num_patents:
                break

            # Find and click the 'Next' button
            try:
                # MODIFICATION: Find the link specifically by its visible text, "Next".
                next_button = driver.find_element(By.LINK_TEXT, "Next")
                
                # Use JavaScript click to avoid interception issues
                driver.execute_script("arguments[0].click();", next_button)
                page_count += 1
                logger.info("Navigating to the next page.")
                time.sleep(3) # Wait for the next page to load
            except NoSuchElementException:
                logger.info("No 'Next' button found. Reached the end of results.")
                break
        
        return patents[:num_patents] # Return the exact number requested

    except Exception as e:
        logger.error(f"An error occurred during scraping: {e}")
        return patents
    finally:
        if driver:
            driver.quit()
            logger.info("WebDriver closed.")

def save_to_json(query, data):
    """Saves the scraped data to a JSON file named after the query."""
    # Sanitize the query to create a valid filename
    filename = re.sub(r'[^a-zA-Z0-9_]', '', query.replace(' ', '_')) + ".json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved {len(data)} patents to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to save data to JSON file: {e}")

def main():
    """Main function to run the patent scraper."""
    print("Google Scholar Patent Scraper")
    print("=============================")
    
    query = input("Enter your patent search query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    scraped_patents = scrape_google_patents(query, num_patents=50)

    if scraped_patents:
        save_to_json(query, scraped_patents)
    else:
        logger.warning("Scraping did not return any patents.")

if __name__ == "__main__":
    main()
