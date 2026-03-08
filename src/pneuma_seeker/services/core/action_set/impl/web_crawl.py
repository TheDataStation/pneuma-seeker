from pneuma_seeker.shared.schemas.core.action import ActionNames
from pneuma_seeker.services.core.action_set.interfaces import Action


class WebCrawl(Action):
    def get_name(self) -> str:
        return ActionNames.WEB_CRAWL.value

    def get_description(self) -> str:
        return f"""**{ActionNames.WEB_CRAWL.value}**
    - Crawls a specified web page to extract textual content for table materialization.
    - Args: {{"url": "<URL of the web page to crawl>"}}
    - Usage notes:
        - Use this when the user specifically requests information from a particular URL.
        - The crawler respects robots.txt and will not fetch disallowed paths.
        - Returned content is raw extracted text from the page (no summarization)."""

    def get_input_schema(self) -> dict[str, str]:
        return {
            "prompt": "The input query to crawl the web.",
            "k": "The number of results to retrieve.",
            "sample_only": "Whether to sample the results only.",
            "sample_size": "The size of the sample if sample_only is True.",
        }

    def get_notes(self) -> str:
        return "This action uses the IR system to crawl the web."
