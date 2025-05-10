from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# ✅ Set up Chrome options
chrome_options = Options()
chrome_options.headless = False  # Set to True if you want it headless
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# ✅ Initialize WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

main_url = "https://iprsearch.ipindia.gov.in/PublicSearch/"

try:
    print("🔹 Opening main page...")
    driver.get(main_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(3)

    search_title = input("Enter a title to search: ").strip()
    search_box = driver.find_element(By.ID, "TextField1")
    search_box.clear()
    search_box.send_keys(search_title)
    search_button = driver.find_element(By.NAME, "submit")
    search_button.click()

    # ✅ CAPTCHA (manual step)
    try:
        captcha_img = driver.find_element(By.ID, "Captcha")
        captcha_url = captcha_img.get_attribute("src")
        print(f"🔹 CAPTCHA detected. Open this URL and solve it manually: {captcha_url}")
        captcha_text = input("Enter the CAPTCHA text: ").strip()

        captcha_input = driver.find_element(By.ID, "CaptchaText")
        captcha_input.send_keys(captcha_text)
        submit_button = driver.find_element(By.NAME, "submit")
        submit_button.click()
        time.sleep(3)

        if "Incorrect CAPTCHA" in driver.page_source:
            print("❌ CAPTCHA verification failed! Try again.")
            driver.quit()
            exit()
        else:
            print("✅ CAPTCHA verified successfully!")
    except:
        print("No CAPTCHA found.")

    results = []
    abstracts = {}

    def extract_current_page():
        print("🔹 Waiting for search results...")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "tr")))
        time.sleep(3)
        page_results = []

        rows = driver.find_elements(By.XPATH, "//table//tr")
        for row in rows:
            try:
                button = row.find_element(By.NAME, "ApplicationNumber")
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 2:
                    app_number = button.get_attribute("value").strip()
                    title = cells[1].text.strip()
                    page_results.append({"application_number": app_number, "title": title, "button": button})
            except:
                continue

        if not page_results:
            print("❌ No results found on page.")
            return []

        # Visit each detail page
        for item in page_results:
            selected_button = item["button"]
            main_window = driver.current_window_handle

            driver.execute_script("arguments[0].click();", selected_button)
            WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) > 1)
            new_window = [w for w in driver.window_handles if w != main_window][0]
            driver.switch_to.window(new_window)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//td/strong[contains(text(),'Abstract')]"))
                )
                abstract_element = driver.find_element(By.XPATH, "//td/strong[contains(text(),'Abstract')]/parent::td")
                abstract_text = abstract_element.text.replace("Abstract:", "").strip()
                abstracts[item["application_number"]] = abstract_text
            except:
                print(f"❌ Abstract not found for {item['application_number']}")
                abstracts[item["application_number"]] = ""

            driver.close()
            driver.switch_to.window(main_window)

        return page_results

    def go_through_all_pages():
        page = 1
        while True:
            print(f"📄 Scraping Page {page}...")
            new_results = extract_current_page()
            results.extend(new_results)
            if page==100:
                break
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'next')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(2)
                page += 1
            except:
                print("🚫 No more pages or 'Next' button not found.")

    go_through_all_pages()

    # ✅ Save results
    serializable_results = [
        {
            "application_number": item["application_number"],
            "title": item["title"],
            "abstract": abstracts.get(item["application_number"], "")
        }
        for item in results
    ]

    with open("../searchformer/patent_search_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(serializable_results)} results to 'patent_search_results.json'")

finally:
    driver.quit()
