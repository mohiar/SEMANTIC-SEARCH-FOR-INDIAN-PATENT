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
    
    
    def _inspect_page_structure(self):
        """Debug method to inspect and log page structure for troubleshooting"""
        try:
            logger.info("=" * 80)
            logger.info("INSPECTING PAGE STRUCTURE FOR DEBUGGING")
            logger.info("=" * 80)
            
            # Check for all select elements
            selects = self.driver.find_elements(By.TAG_NAME, "select")
            logger.info(f"Found {len(selects)} SELECT elements:")
            for i, select in enumerate(selects):
                logger.info(f"  [{i}] id='{select.get_attribute('id')}', "
                           f"name='{select.get_attribute('name')}', "
                           f"class='{select.get_attribute('class')}'")
                options = select.find_elements(By.TAG_NAME, "option")
                for opt in options[:5]:  # Log first 5 options
                    logger.info(f"      - {opt.text}")
            
            # Check for all text input elements
            inputs = self.driver.find_elements(By.XPATH, "//input[@type='text']")
            logger.info(f"Found {len(inputs)} TEXT INPUT elements:")
            for i, inp in enumerate(inputs):
                logger.info(f"  [{i}] id='{inp.get_attribute('id')}', "
                           f"name='{inp.get_attribute('name')}', "
                           f"placeholder='{inp.get_attribute('placeholder')}', "
                           f"class='{inp.get_attribute('class')}'")
            
            # Check for submit buttons
            buttons = self.driver.find_elements(By.XPATH, "//input[@type='submit'] | //button[@type='submit'] | //button")
            logger.info(f"Found {len(buttons)} BUTTON/SUBMIT elements:")
            for i, btn in enumerate(buttons[:10]):  # Log first 10 buttons
                logger.info(f"  [{i}] type='{btn.get_attribute('type')}', "
                           f"value='{btn.get_attribute('value')}', "
                           f"text='{btn.text}', "
                           f"class='{btn.get_attribute('class')}'")
            
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Error inspecting page: {e}")
    
    def search_patents(self, title: str, captcha_value: Optional[str] = None, search_field: str = "Title", max_results: Optional[int] = None) -> Dict:
        """
        Search for patents with given title and optional CAPTCHA value.
        
        Args:
            title: Patent title/query to search
            captcha_value: CAPTCHA value (optional if reusing existing session with valid CAPTCHA)
            search_field: Which field to search (Title, Abstract, Application Number, Complete Specification)
            max_results: Maximum results to extract
        
        Returns:
            Dictionary with results and status
            
        Usage (Pipeline Mode - No CAPTCHA):
            1. Frontend calls /patents/initiate (shows CAPTCHA)
            2. User solves CAPTCHA and calls /patents/search WITH captcha_value
            3. User can call /patents/search again WITHOUT captcha_value (reuses driver + already-validated CAPTCHA)
        """
        try:
            logger.info(f"Starting patent search for: '{title}' in field '{search_field}' (captcha_provided={bool(captcha_value)})")
            # Reset per-search state so consecutive requests do not leak results.
            self.results = []
            self.abstracts = {}
            
            # NOTE: Page is already loaded from get_captcha_screenshot()
            # Do NOT navigate again - reuse existing page with same CAPTCHA shown to user
            time.sleep(2)  # Wait for page stability
            
            # Step 1: Select search field from dropdown
            logger.info(f"Selecting '{search_field}' from dropdown...")
            select_field = self._find_and_select_dropdown(search_field)
            if not select_field:
                raise Exception(f"Could not find or select the '{search_field}' field")
            time.sleep(1)
            
            # Step 2: Enter title
            logger.info(f"Entering search query: {title}")
            search_input = self._find_search_input()
            if not search_input:
                raise Exception("Could not find search input field")
            search_input.clear()
            search_input.send_keys(title)
            time.sleep(1)
            
            # Step 3: Enter CAPTCHA (skip if None - reusing session with already-validated CAPTCHA)
            if captcha_value:
                logger.info("Entering CAPTCHA...")
                captcha_box = self._find_captcha_input()
                if not captcha_box:
                    raise Exception("Could not find CAPTCHA input field")
                captcha_box.clear()
                captcha_box.send_keys(captcha_value)
                time.sleep(1)
            else:
                logger.info("⏭️  Skipping CAPTCHA entry (reusing existing session)")
            
            # Step 4: Click Search button
            logger.info("Clicking search button...")
            search_btn = self._find_search_button()
            if not search_btn:
                raise Exception("Could not find search button")
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
                    "message": "No results found. CAPTCHA may be incorrect or session expired."
                }
        
        except Exception as e:
            logger.error(f"❌ Error during search: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _find_and_select_dropdown(self, option_text: str):
        """Find dropdown using multiple selectors and select option"""
        selectors = [
            "//select[@name='ItemField1']",  # Primary field selector by name
            "//select[contains(@class,'item-select')]",  # Class-based selector
            "//select[contains(@class,'form-control')]",  # Generic form-control
            "//select[@name]",  # Any select with a name
            "//select",  # Last resort - any select
        ]
        
        for selector in selectors:
            try:
                logger.debug(f"Trying selector: {selector}")
                # Wait for element to be visible and clickable (not just present)
                element = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                
                # Give extra time for element to stabilize
                time.sleep(0.5)
                
                # Try to select by visible text
                select_field = Select(element)
                
                # Check if option exists
                try:
                    select_field.select_by_visible_text(option_text)
                    logger.info(f"✅ Found and selected '{option_text}' from dropdown using selector: {selector}")
                    return select_field
                except Exception as opt_err:
                    # If visible text fails, try other approaches
                    logger.debug(f"select_by_visible_text failed: {opt_err}. Trying alternatives...")
                    options = select_field.options
                    for option in options:
                        if option_text.lower() in option.text.lower():
                            option.click()
                            logger.info(f"✅ Selected '{option_text}' using option click method")
                            return select_field
                    raise opt_err
                    
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        
        logger.error("❌ Could not find dropdown with any selector")
        return None
    
    def _find_search_input(self):
        """Find search input field using multiple selectors"""
        selectors = [
            "//input[@name='TextField1']",  # Primary selector by name
            "//input[@placeholder='e.g. COMPUTER IMPLEMENTED']",  # Placeholder match
            "//input[contains(@placeholder, 'COMPUTER')]",  # Partial placeholder match
            "//input[@type='text'][contains(@placeholder, 'e.g.')]",  # Type + placeholder
            "//input[@type='text'][@placeholder]",  # Any text input with placeholder
        ]
        
        for selector in selectors:
            try:
                logger.debug(f"Trying selector: {selector}")
                element = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                logger.info(f"✅ Found search input field using selector: {selector}")
                return element
            except Exception as e:
                logger.debug(f"Search input selector '{selector}' failed: {e}")
                continue
        
        logger.error("❌ Could not find search input field")
        return None
    
    def _find_captcha_input(self):
        """Find CAPTCHA input field using multiple selectors"""
        selectors = [
            "//input[@id='CaptchaText']",  # Primary selector by id
            "//input[@name='CaptchaText']",  # Selector by name
            "//input[@placeholder='Enter Captcha']",  # Exact placeholder match
            "//input[contains(@placeholder, 'Captcha')]",  # Case-sensitive partial match
            "//input[contains(@placeholder, 'captcha')]",  # Case-insensitive partial match
        ]
        
        for selector in selectors:
            try:
                logger.debug(f"Trying selector: {selector}")
                element = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                logger.info(f"✅ Found CAPTCHA input field using selector: {selector}")
                return element
            except Exception as e:
                logger.debug(f"CAPTCHA input selector '{selector}' failed: {e}")
                continue
        
        logger.error("❌ Could not find CAPTCHA input field")
        return None
    
    def _find_search_button(self):
        """Find search button using multiple selectors"""
        selectors = [
            "//input[@type='submit'][@value='Search']",  # Primary: input submit with Value=Search
            "//input[@type='submit' and @value='Search']",  # Alternative syntax
            "//input[@type='submit']",  # Any submit button
            "//button[contains(text(), 'Search')]",  # Button with Search text
            "//button[@type='submit']",  # Any submit button element
        ]
        
        for selector in selectors:
            try:
                logger.debug(f"Trying selector: {selector}")
                element = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                logger.info(f"✅ Found search button using selector: {selector}")
                return element
            except Exception as e:
                logger.debug(f"Search button selector '{selector}' failed: {e}")
                continue
        
        logger.error("❌ Could not find search button")
        return None
    
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
