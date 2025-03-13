import selenium
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import re 


"""Resources:
https://github.com/microsoft/vscode-python/issues/24422
Scroll down to an element: https://www.youtube.com/watch?v=HIstM0Rlt_c&ab_channel=AutomatewithPython
"""
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  Disable 
options.add_argument("--enable-javascript")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')


#********************************************* GET LINKS FROM EACH BLOG SECTION ON X  ***********************************# 


def scrape_x_blogs(link):
    driver = webdriver.Chrome(options=options)
    driver.get(link)
    driver.implicitly_wait(5)
    
    print("Page title:", driver.title)
    time.sleep(3)
    
    count = 0 
    page_height = driver.execute_script("return document.body.scrollHeight")
    back_to_btn = (page_height * 0.70)
    print(page_height)
    print(back_to_btn)
    while count < 8: 
        count += 1
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        scroll_to = new_height - back_to_btn 
        print(f'New Height: {new_height}')
        driver.execute_script('window.scrollTo(0, arguments[0]);', scroll_to)
        try: 
            load_more_button = driver.find_element(By.XPATH, '//*[@id="component-wrapper"]/div[3]/div/div/div[2]/div/div/div/a')
            load_more_button.click()
            print("Clicked!")
        except Exception as e: 
                print("Error clicking button.", e)
                break
    
    parser = BeautifulSoup(driver.page_source, "html.parser")
    header_section = parser.find("div", class_="container--mini container--mobile results-loop")
    article_section = header_section.find_all("div", class_="result__copy")
    article_lst = []
    for article in article_section: 
        a_tag = article.find("a", class_="type--bold-24 color--neutral-dark-gray--has-hover result__title")
        article_lst.append((a_tag.text, "https://blog.x.com" + a_tag['href']))
    
    return article_lst 

section_dct = {"product": "https://blog.x.com/en_us/topics/product", "company":"https://blog.x.com/en_us/topics/company",
       "insights": "https://blog.x.com/en_us/topics/insights", "event": "https://blog.x.com/en_us/topics/events"}

def url_metadata_to_df(url):
    scraped_data = scrape_x_blogs(url)
    section_df = pd.DataFrame(scraped_data, columns = ['Article_Title', 'Link'])
    return section_df 


product_df = url_metadata_to_df(section_dct["product"])
company_df = url_metadata_to_df(section_dct["company"])
insights_df = url_metadata_to_df(section_dct["insights"])
event_df = url_metadata_to_df(section_dct["event"])



#********************************************** SCRAPE EACH ARTICLE FOR TEXT ***********************************# 


def scrape_articles_html(df):
    driver = webdriver.Chrome(options=options)
    for index, row in df.iterrows(): 
        driver.implicitly_wait(1)
        driver.get(row['Link'])    
        try: 
            parser = BeautifulSoup(driver.page_source, "html.parser")

            article_section = parser.find_all("div", class_="bl13-rich-text-editor")
            cleaned_text = [para.get_text(strip=True) for section in article_section for para in section.find_all("p")]
            plain_text = " ".join(cleaned_text)

            date_tag = parser.find("span", class_="b02-blog-post-no-masthead__date color--neutral-light-gray type--roman-14")

            if not date_tag:
                date_tag = parser.find("div", class_="bl01-blog-post-masthead__date type--roman-14 color--neutral-white")
            
            date_text = date_tag.get_text(strip=True) if date_tag else "Date not found!"
 
            df.at[index, "Text"] = plain_text
            df.at[index, "Date"] = date_text
        except Exception as e: 
            print(f"Didn't append text, {e}, error at {index} {row}")
    return df 
    

product_df = scrape_articles_html(product_df)
company_df = scrape_articles_html(company_df)
insights_df = scrape_articles_html(insights_df)
event_df = scrape_articles_html(event_df)

product_df.isnull().sum()
company_df.isnull().sum()
insights_df.isnull().sum()
event_df.isnull().sum()

df_lst = [product_df, company_df, insights_df, event_df]


# Drop any rows that has 2017 dates 

for df in df_lst: 
    len(df)
    drop_indx = []
    for index, data in df.iterrows():
        date = data['Date']
        pattern = "2017"
        match = re.findall(pattern, date)
        if match: 
            drop_indx.append(index)
    
    df.drop(index=drop_indx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    len(df)
            

product_df.to_excel("twitter_product.xlsx", index=False)  
company_df.to_excel("twitter_company.xlsx", index=False)
insights_df.to_excel("twitter_insights.xlsx", index=False)
event_df.to_excel("twitter_events.xlsx", index=False)




