from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class YouTube:
    def __init__(self) -> None:
        # instantiate a chrome options object so you can set the size and headless preference
        self.chrome_options = webdriver.ChromeOptions()
        # self.chrome_options.add_argument("--headless")  
        # Ensure GUI is off. If you want to see the GUI, comment this line.
        # self.driver = webdriver.Chrome(executable_path='/Users/joshrubin/VirtualAssistant/data/Selenium/driver/mac/chromedriver', options=self.chrome_options)



        self.chrome_options.add_experimental_option("useAutomationExtension", False)
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_argument("--disable-notifications")

        self.driver = webdriver.Chrome(options=self.chrome_options)
    def play(self, search_query:str, channel_name:str = None) -> None:
        search_query = search_query.replace(" ", "+") 

        if channel_name is not None:
            search_query += f"+channel:{channel_name.replace(' ', '+')}"
            
        self.driver.get(f'https://youtube.com/search?q={search_query}&sp=EgIQAQ%253D%253D')

        # You can use the WebDriverWait to wait until the element is found or a timeout occurs
        wait = WebDriverWait(self.driver, 10)

        # Specify the xpath
        xpath = '/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[1]/div[1]/ytd-thumbnail/a'

        try:
            # wait until the element is found
            element = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
            # Once the element is found, you can extract the href attribute
            href = element.get_attribute('href')

            # navigate to the video link
            self.driver.get(href)  
        except:
            print("Element not found or timeout")

    def quit(self):
        self.driver.quit()  # Don't forget to quit!

