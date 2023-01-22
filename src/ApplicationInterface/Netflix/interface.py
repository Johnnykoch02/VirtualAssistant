# This file contains the Netflix class used in main.py

import processes
import my_secrets

class Netflix:

    def __init__(self):
        # Create driver
        self.driver = processes.create_driver()

        # Login processes
        processes.login(self.driver, my_secrets.MY_EMAIL, my_secrets.MY_PASSWORD)

    # Watch a show given a search query
    def watch(self, query):
        processes.watch(self.driver, query)

    # Closes the browser
    def quit(self):
        self.driver.quit()