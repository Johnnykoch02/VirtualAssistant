# This file contains the Netflix class used in main.py
import os
import src.ApplicationInterface.Netflix.processes as processes
import time as t
from src.utils import *
class Netflix:

    def __init__(self):
        # Create driver
        self._config = get_json_variables(os.path.join(os.getcwd(), "data", "Netflix", "NetflixConfig.json"), ["EMAIL", "PASSWORD", "PREFERRED_USER"])
        self.driver = processes.create_driver()

        # Login processes
        try:
            processes.login(self.driver, self._config["EMAIL"], self._config["PASSWORD"], self._config["PREFERRED_USER"])
        except:
            processes.login(self.driver, self._config["EMAIL"], self._config["PASSWORD"], self._config["PREFERRED_USER"])
        
        self._prev_is_main_context = True
            

    # Watch a show given a search query
    def watch(self, query):
        processes.watch(self.driver, query)
    
    def run(self, data, is_main_context =False):
        if not is_main_context and self._prev_is_main_context:
            # Context Switch, pause content and resume once its re-opened.
            pass
        elif is_main_context and not self._prev_is_main_context:
            # Resume content from previous point.
            pass
        
        t.sleep(0.1) # Add Sleep for Threading
        self._prev_is_main_context = is_main_context
        
    # Closes the browser
    def quit(self):
        self.driver.quit()