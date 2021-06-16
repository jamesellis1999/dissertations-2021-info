from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time

# specify driver and webpage

chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
driver = webdriver.Chrome(executable_path=chrome_driver)
driver.maximize_window()
driver.get("https://techcrunch.com/startups/")

# wait to load elements on the page
driver.implicitly_wait(5)
# accept cookies, needed every time
element = driver.find_element_by_name("agree")
ActionChains(driver).click(element).perform()
driver.implicitly_wait(3)

data = [] # empty list; will contain URLs to individual pages

number_of_links = 5000 # change this for more articles
articles_per_load = 20

for i in range(0, number_of_links//articles_per_load):
    # rerun current iteration of loop 10 times max if something goes wrong
    # if i%20== 0:
    #     time.sleep(4)
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    j = 0
    while j<10:
        j += 1
        try:
            print(i, end=" ")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            element = driver.find_element_by_class_name("load-more ")
            actions = ActionChains(driver)
            actions.move_to_element(element).perform()
            ActionChains(driver).click(element).perform()
            driver.implicitly_wait(2)
            print("")
            break
        except Exception:
            pass

# extract links
elements = driver.find_elements_by_class_name("post-block__title__link")
for element in elements:
    # print(element.get_attribute('href')) # prints the link for each article's <a>
    data.append(element.get_attribute('href'))

# store links locally
data = pd.DataFrame(data, columns=['links'])
# try to store every 5s until successful
while True:
    try:
        time.sleep(5)
        data.to_csv('techcrunch_links.csv', encoding='utf-8', index=False)
        break
    except:
        print("Error storing file...")
        pass

# close the browser after 45 seconds
driver.implicitly_wait(45)
driver.close()