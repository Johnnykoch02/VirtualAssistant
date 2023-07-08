import os
import src.ApplicationInterface.Hulu.processes as processes
from src.utils import *

class Hulu:

    def __init__(self):
        # Create driver
        self._config = get_json_variables(os.path.join(os.getcwd(), "data", "Hulu", "HuluConfig.json"), ["EMAIL", "PASSWORD", "PREFERRED_USER"])
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