import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from pneuma_seeker.shared.schemas.core.ir_system import (
    AbstractDocument,
    RetrieverType,
    Text,
)
from pneuma_seeker.services.core.ir_system.retriever.interface import (
    AbstractRetriever,
)


class WebCrawler(AbstractRetriever):
    """Represents a web crawler interface."""

    def __init__(self, user_id, chat_id, config, db_api, language_model_api):
        super().__init__(user_id, chat_id, config, db_api, language_model_api)
        self.max_chars = config.WEB_CRAWL_MAX_CHARS
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WebCrawler"})

    @property
    def retriever_type(self) -> RetrieverType:
        """
        Defines the type of the retriever.
        """
        return RetrieverType.WEB_CRAWL

    def load(self):
        """
        Loads the retriever, including its dependencies (e.g., its model).
        """
        pass

    def retrieve(
        self,
        query: str,
        k: int,
        sample_only: bool,
        sample_size: int | None = None,
    ) -> list[AbstractDocument]:
        """
        Crawls the content of the page with URL query (if allowed by its robots.txt).
        """
        if not self.__is_allowed(query):
            return [
                Text(
                    "web_crawl",
                    RetrieverType.WEB_CRAWL,
                    f"Access to {query} is disallowed by robots.txt.",
                    {},
                )
            ]

        try:
            web_content = self.__fetch_content(query)
        except Exception as e:
            web_content = f"Error fetching content from {query}: {e}"
        return [Text("web_crawl", RetrieverType.WEB_CRAWL, web_content, {})]

    def __get_robots_url(self, url: str) -> str:
        """Constructs robots.txt URL based on the given page URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def __is_allowed(self, url: str) -> bool:
        """Checks if the crawler is allowed to fetch the URL according to robots.txt."""
        robots_url = self.__get_robots_url(url)
        try:
            resp = self.session.get(robots_url, timeout=5)
            if resp.status_code != 200:
                return True  # No robots.txt or inaccessible -> assume allowed
            disallowed = [
                line.split(":")[1].strip()
                for line in resp.text.splitlines()
                if line.lower().startswith("disallow:")
            ]
            path = urlparse(url).path
            return not any(path.startswith(d) for d in disallowed if d)
        except requests.RequestException:
            return True  # Fail open: if cannot fetch robots.txt, assume allowed

    def __fetch_content(self, url: str) -> str:
        """Downloads page HTML and extract readable text."""
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[: self.max_chars]

    def index(self, documents: list[AbstractDocument]):
        """
        Indexes a list of documents to the retriever.
        """
        pass
