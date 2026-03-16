#!/usr/bin/env python3
"""
Patent Scraper Service
=====================
Handles patent scraping with CAPTCHA support via web interface
"""

import json
import logging
import os
import time
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatentScraperService:
    """Service to scrape Indian Patent Office patents"""
    
    def __init__(self):
        self.driver = None
        self.results = []
        self.abstracts = {}
    
    def init_driver(self):
        """Initialize Selenium WebDriver"""
        try:
            options = webdriver.ChromeOptions()
            headless = os.getenv("PATENT_SCRAPER_HEADLESS", "true").lower() in {"1", "true", "yes"}
            if headless:
                # Headless mode is safer for server deployments without a display.
                options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            self.driver = webdriver.Chrome(options=options)
            logger.info("✅ WebDriver initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing WebDriver: {e}")
            return False
    
    def _locate_captcha_image(self):
        """Find CAPTCHA image using multiple selectors (main doc + iframes)."""
        selectors = [
            "//img[contains(translate(@alt,'CAPTCHA','captcha'),'captcha')]",
            "//img[contains(translate(@id,'CAPTCHA','captcha'),'captcha')]",
            "//img[contains(translate(@src,'CAPTCHA','captcha'),'captcha')]",
            "//img[contains(translate(@class,'CAPTCHA','captcha'),'captcha')]",
        ]

        # Try in main document first
        for selector in selectors:
            elements = self.driver.find_elements(By.XPATH, selector)
            if elements:
                return elements[0]

        # Then try inside iframes if present
        frames = self.driver.find_elements(By.TAG_NAME, "iframe")
        for frame in frames:
            try:
                self.driver.switch_to.frame(frame)
                for selector in selectors:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        return elements[0]
            except Exception:
                pass
            finally:
                self.driver.switch_to.default_content()

        return None

    def get_captcha_screenshot(self) -> Optional[str]:
        """
        Navigate to patent search page and capture CAPTCHA screenshot.
        Returns base64 encoded image string.
        """
        try:
            logger.info("Navigating to Indian Patent Office search page...")
            self.driver.get("https://iprsearch.ipindia.gov.in/PublicSearch/")
            time.sleep(3)
            import base64

            # Give the page a chance to render dynamic CAPTCHA.
            WebDriverWait(self.driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            captcha_img = self._locate_captcha_image()

            if captcha_img:
                logger.info("✅ CAPTCHA located, capturing element screenshot...")
                return base64.b64encode(captcha_img.screenshot_as_png).decode('utf-8')

            # Fallback: return full-page screenshot so UI still receives an image.
            logger.warning("⚠️ CAPTCHA element not found, returning full-page screenshot fallback")
            screenshot = self.driver.get_screenshot_as_png()
            return base64.b64encode(screenshot).decode('utf-8')
        
        except Exception as e:
            logger.error(f"❌ Error getting CAPTCHA screenshot: {e}")
            return None
    
    def search_patents(self, title: str, captcha_value: str, max_results: Optional[int] = None) -> Dict:
        """
        Search for patents with given title and CAPTCHA value.
        
        Args:
            title: Patent title to search
            captcha_value: CAPTCHA value entered by user
        
        Returns:
            Dictionary with results and status
        """
        try:
            logger.info(f"Starting patent search for: '{title}'")
            # Reset per-search state so consecutive requests do not leak results.
            self.results = []
            self.abstracts = {}
            
            # Step 1: Select "Title" in dropdown
            logger.info("Selecting 'Title' from dropdown...")
            select_field = Select(self.driver.find_element(
                By.XPATH, "//select[contains(@class,'item-select')]"
            ))
            select_field.select_by_visible_text("Title")
            time.sleep(1)
            
            # Step 2: Enter title
            logger.info(f"Entering title: {title}")
            search_input = self.driver.find_element(
                By.XPATH, "//input[@placeholder='e.g. COMPUTER IMPLEMENTED']"
            )
            search_input.clear()
            search_input.send_keys(title)
            time.sleep(1)
            
            # Step 3: Enter CAPTCHA
            logger.info("Entering CAPTCHA...")
            captcha_box = self.driver.find_element(
                By.XPATH, "//input[@placeholder='Enter Captcha']"
            )
            captcha_box.clear()
            captcha_box.send_keys(captcha_value)
            time.sleep(1)
            
            # Step 4: Click Search button
            logger.info("Clicking search button...")
            search_btn = self.driver.find_element(
                By.XPATH, "//input[@type='submit' and @value='Search']"
            )
            search_btn.click()
            time.sleep(3)
            
            # Step 5: Check for results
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table//tr"))
                )
                logger.info("✅ Search results loaded")
                
                # Extract results from current page
                self._extract_page_results(max_results=max_results)

                # Continue pagination until limit reached or pages end
                while True:
                    if max_results and len(self.results) >= max_results:
                        break
                    moved = self._try_next_page(max_results=max_results)
                    if not moved:
                        break
                
                return {
                    "status": "success",
                    "total_results": len(self.results),
                    "results": self._serialize_results()
                }
            
            except TimeoutException:
                logger.error("❌ No results found or CAPTCHA incorrect")
                return {
                    "status": "error",
                    "message": "No results found. CAPTCHA may be incorrect."
                }
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _extract_page_results(self, max_results: Optional[int] = None):
        """Extract patent data from current page"""
        try:
            logger.info("Extracting results from page...")
            rows = self.driver.find_elements(By.XPATH, "//table//tr")
            
            page_results = []
            for row in rows:
                try:
                    button = row.find_element(By.NAME, "ApplicationNumber")
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) >= 2:
                        app_number = button.get_attribute("value").strip()
                        title = cells[1].text.strip()
                        page_results.append({
                            "application_number": app_number,
                            "title": title,
                            "button": button
                        })
                except NoSuchElementException:
                    continue
            
            logger.info(f"Found {len(page_results)} patents on page")
            
            # Extract abstracts for each patent
            for i, record in enumerate(page_results, 1):
                if max_results and len(self.results) >= max_results:
                    logger.info(f"Reached max_results={max_results}, stopping extraction")
                    break
                logger.info(f"Processing patent {i}/{len(page_results)}: {record['title'][:50]}...")
                self._extract_abstract(record)
                self.results.append(record)
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"❌ Error extracting page results: {e}")
    
    def _extract_abstract(self, record: Dict):
        """Extract abstract by opening patent detail page"""
        try:
            main_window = self.driver.current_window_handle
            
            # Click the patent link
            self.driver.execute_script("arguments[0].click();", record["button"])
            WebDriverWait(self.driver, 10).until(lambda d: len(d.window_handles) > 1)
            
            # Switch to new window
            new_window = [w for w in self.driver.window_handles if w != main_window][0]
            self.driver.switch_to.window(new_window)
            
            try:
                # Wait for abstract
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//td/strong[contains(text(),'Abstract')]"))
                )
                
                abstract_element = self.driver.find_element(
                    By.XPATH, "//td/strong[contains(text(),'Abstract')]/parent::td"
                )
                abstract_text = abstract_element.text.replace("Abstract:", "").strip()
                self.abstracts[record["application_number"]] = abstract_text
                logger.info(f"✅ Extracted abstract for {record['application_number']}")
            
            except TimeoutException:
                self.abstracts[record["application_number"]] = "Abstract not available"
                logger.warning(f"⚠️  Abstract not found for {record['application_number']}")
            
            # Close window and return to main
            self.driver.close()
            self.driver.switch_to.window(main_window)
        
        except Exception as e:
            logger.error(f"Error extracting abstract: {e}")
            self.abstracts[record["application_number"]] = "Error extracting abstract"
    
    def _try_next_page(self, max_results: Optional[int] = None):
        """Try to navigate to next page"""
        if max_results and len(self.results) >= max_results:
            return False
        try:
            logger.info("Looking for next page...")
            next_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'next')]"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            self.driver.execute_script("arguments[0].click();", next_button)
            logger.info("🔹 Navigating to next page...")
            time.sleep(2)
            
            # Extract from next page
            self._extract_page_results(max_results=max_results)
            return True
        
        except:
            logger.info("ℹ️  No next page found or only one page of results")
            return False
    
    def _serialize_results(self) -> List[Dict]:
        """Convert results to serializable format"""
        return [
            {
                "application_number": r["application_number"],
                "title": r["title"],
                "abstract": self.abstracts.get(r["application_number"], "")
            }
            for r in self.results
        ]
    
    def save_results(self, filename: str = "patent_search_results.json") -> bool:
        """Save results to JSON file"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self._serialize_results(), f, indent=4, ensure_ascii=False)
            logger.info(f"✅ Saved {len(self.results)} patents to {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            return False
    
    def close(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("✅ WebDriver closed")
