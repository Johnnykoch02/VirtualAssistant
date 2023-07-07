import pickle
import spotipy
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from spotipy.oauth2 import SpotifyOAuth
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import src.utils as utils
from enum import Enum

class Spotify:

    class StatusCode(Enum):
        OK = 200
        CREATED = 201
        ACCEPTED = 202
        BAD_REQUEST = 400
        UNAUTHORIZED = 401
        FORBIDDEN = 403
        NOT_FOUND = 404
        SERVER_ERROR = 500
        PLAY = 100
        PAUSE = 101
        STOP = 102

    def __init__(self):
        #Import keys from cred.json file
        self.data_path = os.path.join(os.getcwd(),'data', 'Spotify', 'SpotifyConfig.json')
        self.config_data = utils.get_json_variables(self.data_path, ['client_id', 'client_secret', 'redirect_uri', 'email', 'password'])

        #spotipy API object
        self._scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing user-read-recently-played streaming user-library-read user-read-private user-library-modify user-read-playback-position playlist-read-private playlist-modify-private"
        self._sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=self.config_data['client_id'], client_secret=self.config_data['client_secret'], redirect_uri=self.config_data['redirect_uri'], scope=self._scope))

    def checkCookies(self):
        try: 
            pickle.load(open('cookies.pkl', 'rb'))
            return Spotify.StatusCode.OK.value
        except:   
            return Spotify.StatusCode.NOT_FOUND.value
    
    def webBrowser(self, value):
        self.value = value
        if value==Spotify.StatusCode.OK.value:
            #chromedriver as environment variable
            self.webdriver_service = Service(os.getenv('chromedriver'))
            
            #chromedriver options
            self.chrome_options=Options()
            self.chrome_options.add_argument("--no-sandbox")
            self.chrome_options.add_experimental_option("detach", True)
            
            #option to remove error: Failed to read descriptor from node connection: A device attached to the system is not functioning
            self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

            self.browser = webdriver.Chrome(service=self.webdriver_service, options=self.chrome_options)
            return Spotify.StatusCode.OK.value
        
        #chromedriver as environment variable
        self.webdriver_service = Service(os.getenv('chromedriver'))
        
        #chromedriver options
        self.chrome_options=Options()
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_experimental_option("detach", True)
        self.chrome_options.add_argument("--headless")
        
        #option to remove error: Filed to read descriptor from node connection: A device attached to the system is not functioning
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        #create browser
        self.browser = webdriver.Chrome(service=self.webdriver_service, options=self.chrome_options)
        return Spotify.StatusCode.OK.value

    def createCookies(self):
        #browser for cookies dump
        Spotify.webBrowser(self,Spotify.StatusCode.NOT_FOUND.value)

        #open spotify on a new headless browser
        self.browser.get(r"https://accounts.spotify.com/en/login?continue=https%3A%2F%2Fopen.spotify.com%2F")

        #wait until page is loaded
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="login-button"]/div[1]')))
            #print('page ready')
        except TimeoutException:
            return Spotify.StatusCode.SERVER_ERROR.value

        #insert email
        email = self.browser.find_element(By.XPATH, '//*[@id="login-username"]')
        for c in self.config_data['email']:
            email.send_keys(c)

        #insert password
        password = self.browser.find_element(By.XPATH, '//*[@id="login-password"]')
        for c in self.config_data['password']:
            password.send_keys(c)

        #click login button
        logbutton = self.browser.find_element(By.XPATH, '//*[@id="login-button"]/div[1]')
        logbutton.click()

        #wait for page to load
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[2]/div[2]/footer/div/div[2]/div/div[1]/button')))
            #print('page ready')
        except TimeoutException:
            return Spotify.StatusCode.UNAUTHORIZED.value
        
        #cookies creation
        cookies=self.browser.get_cookies()
        pickle.dump(cookies , open("cookies.pkl","wb"))

        self.browser.quit()

        return Spotify.StatusCode.CREATED  
        
    def loadCookies(self):
        Spotify.webBrowser(self,Spotify.StatusCode.OK.value)
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

    def openWebPlayer(self):
        #open web player
        self.browser.get('https://open.spotify.com')
        self.browser.maximize_window()

        #wait for page to load
        try:
            WebDriverWait(self.browser, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="main"]/div/div[2]/nav/div[1]/ul/li[1]/a')))
            #print('page ready')
        except TimeoutException:
            return Spotify.StatusCode.SERVER_ERROR.value
        
        #clicking the home button (link) to activate
        (self.browser.find_element(By.CLASS_NAME, 'eNs6P3JYpf2LScgTDHc6')).click()

        return Spotify.StatusCode.ACCEPTED.value
    
    def setup(self):
        if Spotify.checkCookies(self)==Spotify.StatusCode.OK.value:
            Spotify.loadCookies(self)
            Spotify.openWebPlayer(self)
        elif Spotify.checkCookies(self)==Spotify.StatusCode.NOT_FOUND.value:
            Spotify.createCookies(self)
            Spotify.loadCookies(self)
            Spotify.openWebPlayer(self)
            #clicking the home button (link) to activate
            (self.browser.find_element(By.CLASS_NAME, 'eNs6P3JYpf2LScgTDHc6')).click()

    def getDevice(self):
        #get device id
        devices=self._sp.devices()
        device_id=devices['devices'][0]['id']
        #not needed to transfer playback from one device to another. in case of bugs, try adding it
        #self._sp.transfer_playback(device_id, force_play=False)
        return device_id
    
    #play track
    def play(self):
        self._sp.start_playback()
        return Spotify.StatusCode.PLAY.value

    #pause track
    def pause(self):
        self._sp.pause_playback()
        return Spotify.StatusCode.PAUSE.value
    
    #stop application
    def stop(self):
        try:
            self._sp.pause_playback()
        except:
            pass
        self.browser.quit()
        return Spotify.StatusCode.STOP.value

    #search query function
    def search(self, query):
        
        self.query = query

        #get id
        device_id=Spotify.getDevice(self)

        # Search for the Song.
        searchQuery = query

        #get results
        trackResults = self._sp.search(searchQuery,1,0,"track")
        trackURI=trackResults['tracks']['items'][0]['uri']
        trackSelectedList=[]
        trackSelectedList.append(trackURI)
        self._sp.start_playback(device_id, None, trackSelectedList)
        
        return Spotify.StatusCode.ACCEPTED.value