import pickle
import spotipy
import cred
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from spotipy.oauth2 import SpotifyOAuth
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

class Spotify:


    #chrome driver
    webdriver_service = Service(".\chromedriver.exe")

    #options
    chrome_options=Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_experimental_option("detach", True)
    #option to remove error: Filed to read descriptor from node connection: A device attached to the system is not functioning
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    #create browser
    browser = webdriver.Chrome(service=webdriver_service, options=chrome_options)

    #spotipy object API
    scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing user-read-recently-played streaming user-library-read user-read-private user-library-modify user-read-playback-position playlist-read-private playlist-modify-private"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cred.client_id, client_secret=cred.client_secret, redirect_uri=cred.redirect_uri, scope=scope))

    def __init__(self, user, passw):
        self.user = user
        self.passw = passw

    def setup(self):

        try:
            cookies=pickle.load(open('cookies.pkl', 'rb'))
            return 'cookies already set'
        except:
            pass
        self.chrome_options.add_argument("--headless")

        #open spotify on a new browser
        self.browser.get(r"https://accounts.spotify.com/en/login?continue=https%3A%2F%2Fopen.spotify.com%2F")

        #wait until page is loaded
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="login-button"]/div[1]')))
            #print('page ready')
        except TimeoutException:
            print('loading took too much time')

        #insert email
        email = self.browser.find_element(By.XPATH, '//*[@id="login-username"]')
        for c in self.user:
            email.send_keys(c)

        #insert password
        password = self.browser.find_element(By.XPATH, '//*[@id="login-password"]')
        for c in self.passw:
            password.send_keys(c)

        #click login button
        logbutton = self.browser.find_element(By.XPATH, '//*[@id="login-button"]/div[1]')
        logbutton.click()

        #wait for page to load
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[2]/div[2]/footer/div/div[2]/div/div[1]/button')))
            #print('page ready')
        except TimeoutException:
            print('loading took too much time')
        
        #cookies
        cookies=self.browser.get_cookies()
        #print(cookies)
        pickle.dump(cookies , open("cookies.pkl","wb"))


        self.browser.quit()

        return 'cookies have been set'

    def open(self):
        try:
            self.chrome_options.arguments.remove("--headless")
        except:
            pass
        #open spotify on a new browser
        self.browser.get(r"https://accounts.spotify.com/en/login?continue=https%3A%2F%2Fopen.spotify.com%2F")

        #load cookies
        cookies=pickle.load(open('cookies.pkl', 'rb'))

        for cookie in cookies:
            cookie['domain']='.spotify.com'
            try:
                self.browser.add_cookie(cookie)
            except:
                pass

        #open web player
        self.browser.get('https://open.spotify.com')
        self.browser.maximize_window()

        #wait for page to load
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[2]/nav/div[1]/ul/li[1]/a')))
            #print('page ready')
        except TimeoutException:
            print('loading took too much time')

        #check for announcement and close it
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="onetrust-close-btn-container"]/button')))
            time.sleep(1)
            xbutton = self.browser.find_element(By.XPATH, '//*[@id="onetrust-close-btn-container"]/button')
            xbutton.click()
        except:
            pass

    def search(self, query):
        self.query = query

        #get device id
        devices=self.sp.devices()
        device_id=devices['devices'][0]['id']
        self.sp.transfer_playback(device_id, force_play=False)

        # Search for the Song.
        if(query=='stop'):
            self.sp.pause_playback()
            self.browser.quit()
            return 'Spotify has closed'
        if(query=='pause'):
            self.sp.pause_playback()
            return 'Playback paused'

        elif(query=='play'):
            self.sp.start_playback()
            return 'Playback resumed'
        else:
            searchQuery = query
            #get results
            trackResults = self.sp.search(searchQuery,1,0,"track")
            trackURI=trackResults['tracks']['items'][0]['uri']
            trackSelectedList=[]
            trackSelectedList.append(trackURI)
            #sp.transfer_playback(device_id, force_play=True)
            #print(json.dumps(devices, sort_keys=True, indent=4))
            self.sp.start_playback(device_id, None, trackSelectedList)
            print(self.sp.current_user_playing_track())
            return 


        

ale = Spotify(cred.user, cred.passw)

print(ale.setup())

ale.open()
q=input("search for song (press stop to close spotify): ")
ale.search(q)