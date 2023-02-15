# This file contains the Netflix class used in main.py

# import processes
import src.ApplicationInterface.Netflix.my_secrets as my_secrets
# import json
import os
import src.ApplicationInterface.Netflix.processes as processes

from src.utils import *

# Define Chrome driver path

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
            

    # Watch a show given a search query
    def watch(self, query):
        processes.watch(self.driver, query)

    # Closes the browser
    def quit(self):
        self.driver.quit()