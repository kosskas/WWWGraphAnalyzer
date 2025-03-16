import sympy
import time
from Robot import WWWRobot

if __name__ == "__main__":
    start_url = "https://pg.edu.pl"
    start_time = time.time()
    scraper = WWWRobot(start_url, 1000)
    scraper.start()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} s")
