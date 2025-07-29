from Robot import WWWRobot

if __name__ == "__main__":
    start_url = "https://www.manchester.ac.uk"
    scraper = WWWRobot(start_url, 3500, 8)
    scraper.start()
