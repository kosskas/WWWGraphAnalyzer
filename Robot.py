import os
import requests
from bs4 import BeautifulSoup
import threading
from queue import Queue
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from Graph import WWWGraph
import re

DIR = "./download"


class WWWRobot:
    def __init__(self, start_url, pages_limit):
        self.start_url = start_url
        self.visited_pages = set()
        self.pages_limit = pages_limit
        self.queue = Queue()
        self.www_graph = WWWGraph(pages_limit)
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(start_url, "robots.txt"))
        self.robot_parser.read()
        self.idx = 0

    def is_allowed(self, url):
        return self.robot_parser.can_fetch("*", url)

    def get_page(self, parent, url):
        if url not in self.visited_pages and self.is_allowed(url):
            try:
                print(f"getting {url}")
                response = requests.get(url)
                if "text/html" not in response.headers["content-type"]:
                    return None
                if response.status_code == 200:
                    page_content = response.text
                    self.visited_pages.add(url)
                    if self.idx > 0:
                        self.www_graph.addDirectedEdge(parent, url)
                    self.idx += 1
                    return page_content
            except requests.RequestException as e:
                print(f"Error reading page {url}: {e}")
        return None

    def validate_url(self, url):
        url = re.sub(r"#.*", "", url)  # delete subsection
        url = re.sub(r"\?.*", "", url)  # delete get method

        if url[-1] == "/":
            url = url[:-1]
        return url

    def parse_links(self, page_content, url):
        dom = BeautifulSoup(page_content, "html.parser")
        links = set()
        for anchor in dom.find_all("a", href=True):
            link = urljoin(url, anchor["href"])
            if urlparse(link).netloc == urlparse(self.start_url).netloc:
                links.add(self.validate_url(link))
        return links

    def start(self):
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        self.queue.put((None, self.start_url))
        while not self.queue.empty() and self.idx < self.pages_limit:
            print(f"{self.idx}/{self.pages_limit} Crawled")
            parent, url = self.queue.get()
            page = self.get_page(parent, url)
            if page is not None:
                links = self.parse_links(page, url)
                for link in links:
                    if link not in self.visited_pages:
                        self.queue.put((url, link))

        self.www_graph.printGraph()
        print(self.visited_pages)
