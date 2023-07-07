import pickle
import sys
import os
import datetime

from selenium import webdriver
import selenium.webdriver.chrome.options
import selenium.webdriver.firefox.options
import selenium.webdriver.edge.options 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException

# Setup and creation of driver Object for usage in interface.py
def create_driver():

    # Determine the os type to be used in the driver path
    # os_type = "windows" if sys.platform == "win32" else "linux"
    # driver_executable = "chromedriver.exe" if sys.platform == "win32" else "chromedriver"

    os_type = "windows" if sys.platform == "win32" else "mac" if sys.platform == "darwin" else "linux"
    driver_executable = "chromedriver.exe" if sys.platform == "win32" else "chromedriver"



    driver_path = os.path.join(os.getcwd(), "data", "Selenium", "driver", os_type, driver_executable)
    driver_runner = webdriver.Chrome
    
    # Set's up options
    current_options = selenium.webdriver.chrome.options.Options()
    # For disabling the "this browser is being controlled by automated software"
    current_options.add_experimental_option("useAutomationExtension", False)
    current_options.add_experimental_option("excludeSwitches",["enable-automation"])
    current_options.add_argument("--disable-notifications")

    # Creates the driver Object to be used in interface.py
    # try:
    #     driver = driver_runner(
    #         executable_path = driver_path,
    #         options = current_options
    #     )
    try:
        driver = webdriver.Chrome(
            # executable_path = driver_path,
            options = current_options
        )
    except Exception as e:  # Catch any exception
        print(f"Error during driver creation: {e}")  # Print the error message
        raise SystemExit

    driver.maximize_window()

    return driver

# Tries to bypass Netflix login with stored cookies, or log's in and stores new cookies
def login(driver, username, password, preferred_user):
    USERNAME_FIELD = (By.CSS_SELECTOR, 'input[name="userLoginId"]')
    PASSWORD_FIELD = (By.CSS_SELECTOR, 'input[name="password"]')
    BILLBOARD_PLAY_BUTTON = (By.CSS_SELECTOR, 'div.billboard-row  a.playLink')
    cookies_path = os.path.join(os.getcwd(), "data", "Netflix", "pickledcookies.pkl")

    # Try opening pickledcookies.pkl
    try:

        # Get stored cookies and open login page
        with open(cookies_path, 'rb') as pickledcookies:
            browser_settings = pickle.load(pickledcookies)
            driver.delete_all_cookies()
            driver.get('https://netflix.com/login')

            # If cookies are fresh, use them and bypass login
            if browser_settings['last_updated'] == datetime.date.today():
                for cookie in browser_settings['stored_cookies']:
                    driver.add_cookie({k: v for k, v in cookie.items() if k != 'expiry'})
                driver.refresh()

            # This means cookies are not fresh
            else:

                # Do login
                username_field = driver.find_element(*USERNAME_FIELD)
                username_field.send_keys(username)
                password_field = driver.find_element(*PASSWORD_FIELD)
                password_field.send_keys(password)
                password_field.submit()
                """ Wait for the login to fail or succeed
                    Netflix does something interesting here. If you submit invalid 
                    credentials, the login page refreshes. Thus the password_field 
                    will always become stale after each login attempt """
                wait = WebDriverWait(driver, 10)
                wait.until(EC.staleness_of(password_field))

                # Save new cookies
                browser_settings = dict()
                browser_settings['last_updated'] = datetime.date.today()
                browser_settings['stored_cookies'] = driver.get_cookies()
                with open('./pickledcookies.pkl', 'wb') as pickledcookies:
                    pickle.dump(browser_settings, pickledcookies)

    # If pickledcookies.pkl does not exist, create it and raise
    except FileNotFoundError:
        with open(cookies_path, 'wb') as pickledcookies:
            browser_settings = dict()
            browser_settings['last_updated'] = datetime.date(1980,1,1)
            browser_settings['stored_cookies'] = 1234567890
            pickle.dump(browser_settings, pickledcookies)
        print("pickledcookies.pkl did not exist, trying to create it.")
        raise FileNotFoundError

    # After login, we are either in the homepage, or user selection page
    # Wait for either home button or user select header
    wait = WebDriverWait(driver, 10)
    HOME_BUTTON = (By.CSS_SELECTOR, 'a[aria-label="Netflix"]')
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'h1.profile-gate-label , a[aria-label="Netflix"]')))

    
    def safe_find_element(driver, by, value):
        try:
            return driver.find_element(by, value)
        except NoSuchElementException:
            return None
        


    element = safe_find_element(driver, By.CSS_SELECTOR, 'h1.profile-gate-label')
    if element:
    # if driver.find_element(By.CSS_SELECTOR, 'h1.profile-gate-label'):

        # Try logging in to our preferred user, or first user in case of failure
        try:
            # If we cannot find this <span>, then the except statement will trigger
            selector = "//span[text()='%s']" % preferred_user
            span = driver.find_element(By.XPATH, selector)

            # This means that we can find that <span>, so lets use it to select user
            link = span.find_element(By.XPATH, '..')
            link.click()

        # This means we could not find the preferred user, select a default user then
        except (NoSuchElementException, AttributeError) as error:
            print("Could not find Preferred user, selecting first option")
            link = driver.find_element(By.CSS_SELECTOR, 'a[data-uia="action-select-profile+primary"]')
            link.click()

    # Check for a homepage, which means we are good to go
    elif driver.find_element(*BILLBOARD_PLAY_BUTTON):
        pass
    
    # *Should* never happen, but just in case
    else:
        print("Error, could not find a page after logging in")
        raise SystemExit
    
    # Login successful, let's wait for the page to open
    HOME_BUTTON = (By.CSS_SELECTOR, 'a[aria-label="Netflix"]')
    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located(HOME_BUTTON))

# Given a search query, searches for the best match and plays it
def watch(driver, query):

    driver.get("https://www.netflix.com")

    # Let's wait for the home page to open
    HOME_BUTTON = (By.CSS_SELECTOR, 'a[aria-label="Netflix"]')
    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located(HOME_BUTTON))

    # Used page elements
    SEARCH_BUTTON = (By.CSS_SELECTOR, 'button.searchTab')
    SEARCH_FIELD = (By.CSS_SELECTOR, 'input[data-uia="search-box-input"]')

    # Returns true if the search field is open (essentially, if it exists)
    def search_field_is_open(driver):
        try:
            driver.find_element(*SEARCH_FIELD)
            return True
        except NoSuchElementException:
            return False
    # Clears the search field
    def clear_search(driver):
        search_field = driver.find_element(*SEARCH_FIELD)
        search_field.clear()

    # Clear the search field if it's open, else open it by clicking the search button
    if search_field_is_open(driver):
        clear_search(driver)
    else:
        search_button = driver.find_element(*SEARCH_BUTTON)
        search_button.click()

    # Search for our query
    search_field = driver.find_element(*SEARCH_FIELD)
    for char in query:
        search_field.send_keys(char)

    # Wait until the first card loads
    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div#title-card-0-0')))

    # Now we are going to play the first show that appears, regardless of whether it is or not what we want
    # First we open the first card
    first_show = driver.find_element(By.CSS_SELECTOR, 'div#title-card-0-0')
    link = first_show.find_element(By.CSS_SELECTOR, "a")
    link.click()
    # Then we click the play button
    controls = driver.find_element(By.CSS_SELECTOR, 'div[data-uia="mini-modal-controls"]')
    play_button = controls.find_element(By.CSS_SELECTOR, 'button.color-primary')
    play_button.click()

    # All we need to do now is click the fullscreen button
    # But let's wait for that to appear first
    FULL_SCREEN_BUTTON = (By.CSS_SELECTOR, 'button[data-uia="control-fullscreen-enter"]')
    wait = WebDriverWait(driver, 10)
    wait.until(EC.visibility_of_element_located(FULL_SCREEN_BUTTON))

    # Now we click it
    full_screen_button = driver.find_element(*FULL_SCREEN_BUTTON)
    full_screen_button.click()