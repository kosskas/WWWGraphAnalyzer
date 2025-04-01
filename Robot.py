from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
from bs4 import BeautifulSoup
import threading
from queue import Queue
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from Graph import WWWGraph
import re
import sympy
import time
import shutil


class WWWRobot:
    def __init__(self, start_url, pages_limit, max_threads):
        self.start_url = start_url
        self.visited_pages = set()
        self.pages_limit = pages_limit
        self.queue = Queue()
        self.www_graph = WWWGraph(pages_limit)
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(urljoin(start_url, "robots.txt"))
        self.robot_parser.read()
        self.idx = 0
        self.lock = threading.Lock()
        self.max_threads = max_threads
        self.save_dir = "./download"
        self.clear_directory()

    def is_allowed(self, url):
        return self.robot_parser.can_fetch("*", url)

    def clear_directory(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def get_page(self, url):
        if self.is_allowed(url):
            try:
                response = requests.get(url, timeout=5)
                if "text/html" not in response.headers["content-type"]:
                    return None
                if response.status_code == 200:
                    page_content = response.text
                    self.save_page(url, page_content)
                    return page_content
            except requests.RequestException as e:
                print(f"Error reading page {url}: {e}")
        return None

    def format_link(self, url):
        url = (
            re.sub(r"https?://", "", url)
            .replace("/", "-")
            .replace(".", "-")
            .replace("%", "-")
        )
        url = re.sub(r"#.*", "", url)
        url = re.sub(r"\?.*", "", url)

        if url[-1] == "/":
            url = url[:-1]
        return url

    def save_page(self, url, content):
        filename = self.format_link(url) + "html"
        filepath = os.path.join(self.save_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                print(
                    f"[{threading.get_ident()}] [V={self.www_graph.nodes()} E={self.www_graph.edges()}]  saving {url}"
                )
        except Exception as e:
            print(f"Error {filepath}: {e}")

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

    def worker(self):
        while not self.queue.empty():
            self.crawl_page()

    def crawl_page(self):
        parent, url = self.queue.get()

        with self.lock:
            if self.format_link(url) in self.visited_pages:
                self.www_graph.addDirectedEdge(parent, url)
                self.queue.task_done()
                return
            if self.www_graph.nodes() >= self.pages_limit:  # Zmiana z '>' na '>='
                self.queue.task_done()
                return

        page = self.get_page(url)
        if page is not None:
            with self.lock:
                self.visited_pages.add(self.format_link(url))
                self.www_graph.addDirectedEdge(parent, url)

            links = self.parse_links(page, url)
            for link in links:
                self.queue.put((url, link))

        self.queue.task_done()

    def start(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        start_time = time.time()

        self.queue.put((self.start_url, self.start_url))
        self.crawl_page()

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(self.worker) for _ in range(self.max_threads)]
            for future in as_completed(futures):
                future.result()

        end_time = time.time()

        print(f"Execution time: {end_time - start_time:.4f} s")

        with self.lock:
            self.www_graph.save()
            print(f"V={self.www_graph.nodes()} idx={self.idx}")
