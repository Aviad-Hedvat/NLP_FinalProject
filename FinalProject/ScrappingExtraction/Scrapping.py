import os
import re
from urllib.request import urlretrieve

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions

from webdriver_manager.chrome import ChromeDriverManager

URL = "https://supreme.court.gov.il/Pages/fullsearch.aspx"


def scrape_verdicts(trace: bool = True) -> None:
    save_dir = os.path.join(os.getcwd(), '../corpus')

    if trace:
        print('scrape verdicts'
              f'saving location: {save_dir}\n\n')

    service = Service(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(URL)

    frame = driver.find_element(By.XPATH, '//iframe')
    driver.switch_to.frame(frame)

    doc_type = WebDriverWait(driver, 10).until(
        expected_conditions.presence_of_element_located((By.XPATH, '//sc-select[@ng-model="data.Type"]')),
    )

    doc_type.find_element(By.XPATH, './/input').click()

    verdicts_option = WebDriverWait(driver, 10).until(
        expected_conditions.visibility_of_element_located(
            (By.XPATH, './/div[@class="ng-binding ng-scope" and text()="פסק-דין"]')
        )
    )

    driver.execute_script("arguments[0].click()", verdicts_option)

    search_button = driver.find_element(By.XPATH, '//section[@class="search-bottom"]') \
        .find_element(By.XPATH, './/button[@type="submit"]')

    driver.execute_script("arguments[0].click()", search_button)

    if trace:
        print('performed search of recent verdicts')

    WebDriverWait(driver, 60).until(
        expected_conditions.presence_of_element_located(
            (By.XPATH, '//form[@class="results-page ng-pristine ng-valid ng-scope ng-valid-pattern"]')
        )
    )

    window = driver.find_element(By.XPATH, '//div[@class="results-listing"]')

    html_buttons = driver.find_elements(By.XPATH, '//a[@class="file-link html-link"]')

    while len(html_buttons) < 500:
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight;', window)
        html_buttons = driver.find_elements(By.XPATH, '//a[@class="file-link html-link"]')

    html_hrefs = [e.get_attribute('href') for e in html_buttons]
    html_links = [h if h.startswith("https://supremedecisions.court.gov.il")
                  else f"https://supremedecisions.court.gov.il/{h}"
                  for h in html_hrefs]

    if trace:
        print(f'Retrieved {len(html_links)} html links')

    driver.close()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for link in html_links:
        filename = f"{re.findall(r'fileName=.*&', link)[0][9:-1]}.html"
        urlretrieve(link, f'{save_dir}/{filename}')

    if trace:
        print(f'saved {len(html_links)} html files to {save_dir}\n')


scrape_verdicts()
