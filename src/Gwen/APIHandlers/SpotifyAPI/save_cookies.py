import pickle
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import cred


#chrome driver
webdriver_service = Service(".\chromedriver.exe")

#options
chrome_options=Options()
chrome_options.add_argument("--no-sandbox")
#option to remove error: Filed to read descriptor from node connection: A device attached to the system is not functioning
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

#create browser
browser = webdriver.Chrome(service=webdriver_service, options=chrome_options)

#open spotify on a new browser
browser.get(r"https://accounts.spotify.com/en/login?continue=https%3A%2F%2Fopen.spotify.com%2F")

#wait until page is loaded
#time.sleep(100)
try:
    WebDriverWait(browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="login-button"]/div[1]')))
    print('page ready')
except TimeoutException:
    print('loading took too much time')

#insert email
email = browser.find_element(By.XPATH, '//*[@id="login-username"]')
for c in cred.user:
    email.send_keys(c)

#insert password
password = browser.find_element(By.XPATH, '//*[@id="login-password"]')
for c in cred.passw:
    password.send_keys(c)

#click login button
logbutton = browser.find_element(By.XPATH, '//*[@id="login-button"]/div[1]')
logbutton.click()

#wait for page to load
try:
    WebDriverWait(browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[2]/div[2]/footer/div/div[2]/div/div[1]/button')))
    print('page ready')
except TimeoutException:
    print('loading took too much time')
#cookies
cookies=browser.get_cookies()
#print(cookies)
pickle.dump(cookies , open("cookies.pkl","wb"))


browser.quit()

'''''
try:
        clearBtn = browser.find_element(By.XPATH, '//*[@id="main"]/div/div[2]/div[1]/header/div[3]/div/div/div/button')
        clearBtn.click()
    except:
        pass
    #query=input('What song do you want to listen to? (press x to stop) ')
    query="talk"
    if(query=="stop"):
        break;
    elif(query=="play" or query=="pause"):
        play = browser.find_element(By.XPATH, '//*[@id="main"]/div/div[2]/div[2]/footer/div/div[2]/div/div[1]/button')
        play.click()
    else:
        searchSong = browser.find_element(By.XPATH, '//*[@id="main"]/div/div[2]/div[1]/header/div[3]/div/div/form/input')
        for c in query:
            searchSong.send_keys(c)
        time.sleep(3)
        action = ActionChains(browser)
        square = browser.find_element(By.XPATH, '//*[@id="searchPage"]/div/div/section[1]/div[2]/div/div/div/div[4]')
        action.move_to_element(square).perform()
        time.sleep(5)

        
        play = browser.find_element(By.XPATH, '//*[@id="searchPage"]/div/div/section[2]/div[2]/div/div/div/div[2]/div[1]/div/div[1]/div[1]/button')
        play.click()   
    '''