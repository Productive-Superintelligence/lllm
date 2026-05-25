# Exa Search Proxy
# https://docs.exa.ai/reference/getting-started


import datetime as dt
import os
from dataclasses import asdict

from exa_py import Exa

from ... import utils as U
from ..base import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="exa",
    name="Exa Search API",
    description="Exa is a search engine made for AIs. Exa finds the exact content you’re looking for on the web, with three core functionalities: /SEARCH -> Find webpages using Exa’s embeddings-based or Google-style keyword search. /CONTENTS -> Obtain clean, up-to-date, parsed HTML from Exa search results. /FINDSIMILAR -> Based on a link, find and return pages that are similar in meaning.",
)
class ExaProxy(BaseProxy):
    """
    Welcome to Exa

    Exa is a search engine made for AIs.

    Exa finds the exact content you’re looking for on the web, with three core functionalities:

    /SEARCH ->
    Find webpages using Exa’s embeddings-based or Google-style keyword search.

    /CONTENTS ->
    Obtain clean, up-to-date, parsed HTML from Exa search results.

    /FINDSIMILAR ->
    Based on a link, find and return pages that are similar in meaning.
    """

    def __init__(self, cutoff_date: str = None, use_cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=use_cache, **kwargs)
        self.api_key_name = "apikey"
        self.api_key = os.getenv("EXA_API_KEY")
        self.exa = Exa(self.api_key)
        self.cache_name = "EXA_API"

    def _call_api(
        self, url: str, params: dict, endpoint_info: dict, headers: dict
    ) -> dict:
        cache_key = U.create_cache_key(endpoint_info["endpoint"], params)
        cached_response = U.load_cache_by_key(self.cache_name, cache_key)
        if cached_response is not None and self.use_cache:
            return cached_response

        if self.cutoff_date is not None:
            for keyword in ["CrawlDate", "PublishedDate"]:
                shift = None
                endkey = f"end{keyword}"
                startkey = f"start{keyword}"
                if endkey in params:
                    end_date = dt.datetime.strptime(
                        params[endkey], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    if end_date > self.cutoff_date:
                        params[endkey] = self.cutoff_date.strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                        shift = end_date - self.cutoff_date
                else:
                    end_date = self.cutoff_date
                if startkey in params:
                    start_date = dt.datetime.strptime(
                        params[startkey], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    if shift is not None:
                        params[startkey] = (start_date - shift).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                    if start_date >= end_date:
                        params[startkey] = (end_date - dt.timedelta(days=30)).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        )

        endpoint = endpoint_info["endpoint"]
        if endpoint == "search":
            response = self.exa.search_and_contents(
                params["query"],
                type=params["type"],
                category=params["category"],
                num_results=params["numResults"],
                text=True,
                summary={"query": "Main developments"},
                subpages=1,
                subpage_target="sources",
                extras={"links": 1, "image_links": 1},
                start_crawl_date=params["startCrawlDate"],
                end_crawl_date=params["endCrawlDate"],
                start_published_date=params["startPublishedDate"],
                end_published_date=params["endPublishedDate"],
            )
        elif endpoint == "findSimilar":
            response = self.exa.find_similar_and_contents(
                url=params["url"],
                text=True,
                start_crawl_date=params["startCrawlDate"],
                end_crawl_date=params["endCrawlDate"],
                start_published_date=params["startPublishedDate"],
                end_published_date=params["endPublishedDate"],
            )

        elif endpoint == "contents":
            response = self.exa.get_contents(
                urls=params["urls"],
                text=True,
            )

        response_json = {}
        response_json["results"] = [asdict(r) for r in response.results]
        response_json["autopromptString"] = response.autoprompt_string
        response_json["resolvedSearchType"] = response.resolved_search_type
        response_json["autoDate"] = response.auto_date
        response_json["costDollars"] = (
            asdict(response.cost_dollars) if response.cost_dollars else None
        )

        U.save_cache_by_key(self.cache_name, cache_key, response_json)
        return response_json

    ########################################
    ### Search Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search",
        name="Search API",
        description=(
            "The search endpoint lets you intelligently search the web and extract contents from the results. "
            "By default, it automatically chooses between traditional keyword search and Exa’s embeddings-based model, to find the most relevant results for your query."
        ),
        params={
            "query": (str, "AAPL"),
            "useAutoprompt": (bool, True),
            "type": (str, "auto"),
            "category": (str, "company"),
            "numResults": (int, 10),
            "includeDomains": (list, ["arxiv.org", "paperswithcode.com"]),
            "excludeDomains": (list, []),
            "startCrawlDate": (str, "2023-01-01T00:00:00.000Z"),
            "endCrawlDate": (str, "2023-12-31T00:00:00.000Z"),
            "startPublishedDate": (str, "2023-01-01T00:00:00.000Z"),
            "endPublishedDate": (str, "2023-12-31T00:00:00.000Z"),
            "includeText": (list, ["large language model"]),
            "excludeText": (list, ["course"]),
            "highlightsQuery": (str, "Key advancements"),
            "summaryQuery": (str, "Main developments"),
        },
        response={
            # "requestId": "b5947044c4b78efa9552a7c89b306d95",
            "autopromptString": "Heres a link to the latest research in LLMs:",
            "autoDate": "2024-02-08T02:15:42.180Z",
            "resolvedSearchType": "neural",
            "results": [
                {
                    "title": "A Comprehensive Overview of Large Language Models",
                    "url": "https://arxiv.org/pdf/2307.06435.pdf",
                    "publishedDate": "2023-11-16T01:36:32.547Z",
                    "author": "Humza  Naveed, University of Engineering and Technology (UET), Lahore, Pakistan",
                    "score": 0.4600165784358978,
                    "id": "https://arxiv.org/abs/2307.06435",
                    "image": "https://arxiv.org/pdf/2307.06435.pdf/page_1.png",
                    "favicon": "https://arxiv.org/favicon.ico",
                    "text": "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities...",
                    "highlights": ["Such requirements have limited their adoption..."],
                    "highlightScores": [0.4600165784358978],
                    "summary": "This overview paper on Large Language Models (LLMs) highlights key developments...",
                    "subpages": [
                        {
                            "id": "https://arxiv.org/abs/2303.17580",
                            "url": "https://arxiv.org/pdf/2303.17580.pdf",
                            "title": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
                            "author": "Yongliang  Shen, Microsoft Research Asia, Kaitao  Song, Microsoft Research Asia, Xu  Tan, Microsoft Research Asia, Dongsheng  Li, Microsoft Research Asia, Weiming  Lu, Microsoft Research Asia, Yueting  Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn, Zhejiang  University, Microsoft Research Asia, Microsoft  Research, Microsoft Research Asia",
                            "publishedDate": "2023-11-16T01:36:20.486Z",
                            "text": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face Date Published: 2023-05-25 Authors: Yongliang Shen, Microsoft Research Asia Kaitao Song, Microsoft Research Asia Xu Tan, Microsoft Research Asia Dongsheng Li, Microsoft Research Asia Weiming Lu, Microsoft Research Asia Yueting Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn Zhejiang University, Microsoft Research Asia Microsoft Research, Microsoft Research Asia Abstract Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower t",
                            "summary": "HuggingGPT is a framework using ChatGPT as a central controller to orchestrate various AI models from Hugging Face to solve complex tasks. ChatGPT plans the task, selects appropriate models based on their descriptions, executes subtasks, and summarizes the results. This approach addresses limitations of LLMs by allowing them to handle multimodal data (vision, speech) and coordinate multiple models for complex tasks, paving the way for more advanced AI systems.",
                            "highlights": [
                                "2) Recently, some researchers started to investigate the integration of using tools or models in LLMs  ."
                            ],
                            "highlightScores": [0.32679107785224915],
                        }
                    ],
                    "extras": {"links": []},
                }
            ],
            # "searchType": "auto",
            "costDollars": {
                "total": 0.005,
                "breakDown": [
                    {
                        "search": 0.005,
                        "contents": 0,
                        "breakdown": {
                            "keywordSearch": 0,
                            "neuralSearch": 0.005,
                            "contentText": 0,
                            "contentHighlight": 0,
                            "contentSummary": 0,
                        },
                    }
                ],
                "perRequestPrices": {
                    "neuralSearch_1_25_results": 0.005,
                    "neuralSearch_26_100_results": 0.025,
                    "neuralSearch_100_plus_results": 1,
                    "keywordSearch_1_100_results": 0.0025,
                    "keywordSearch_100_plus_results": 3,
                },
                "perPagePrices": {
                    "contentText": 0.001,
                    "contentHighlight": 0.001,
                    "contentSummary": 0.001,
                },
            },
        },
    )
    def search(self, params: dict) -> dict:
        """
        Parameters:
            - query: The query string for the search.
                - string, required
                - Example: "Latest developments in LLM capabilities"
            - useAutoprompt: Autoprompt converts your query to an Exa-style query. Enabled by default for auto search, optional for neural search, and not available for keyword search.
                - boolean, optional, default:true
                - Example: true
            - type: The type of search. Neural uses an embeddings-based model, keyword is google-like SERP. Default is auto, which automatically decides between keyword and neural.
                - enum<string>, optional, default:auto
                - Available options: keyword, neural, auto
                - Example: "auto"
            - category: A data category to focus on.
                - enum<string>, optional
                - Available options: company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report
                - Example: "research paper"
            - numResults: Number of results to return (up to thousands of results available for custom plans)
                - integer, optional, default:10
                - Required range: x <= 100
                - Example: 10
            - includeDomains: List of domains to include in the search. If specified, results will only come from these domains.
                - string[], optional
                - Example: ["arxiv.org", "paperswithcode.com"]
            - excludeDomains: List of domains to exclude from search results. If specified, no results will be returned from these domains.
                - string[], optional
            - startCrawlDate: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled after this date. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-01-01T00:00:00.000Z"
            - endCrawlDate: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled before this date. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-12-31T00:00:00.000Z"
            - startPublishedDate: Only links with a published date after this will be returned. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-01-01T00:00:00.000Z"
            - endPublishedDate: Only links with a published date before this will be returned. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-12-31T00:00:00.000Z"
            - includeText: List of strings that must be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.
                - string[], optional
                - Example: ["large language model"]
            - excludeText: List of strings that must not be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.
                - string[], optional
                - Example: ["course"]
            - highlightsQuery: Custom query to direct the LLM's selection of highlights.
                - string, optional
                - Example: "Key advancements"
            - summaryQuery: Custom query for the LLM-generated summary.
                - string, optional
                - Example: "Main developments"
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="findSimilar",
        name="Find Similar Links API",
        description="Find similar links to the link provided and optionally return the contents of the pages.",
        params={
            "url*": (str, "https://arxiv.org/abs/2307.06435"),
            "numResults": (int, 10),
            "includeDomains": (list, ["arxiv.org", "paperswithcode.com"]),
            "excludeDomains": (list, []),
            "startCrawlDate": (str, "2023-01-01T00:00:00.000Z"),
            "endCrawlDate": (str, "2023-12-31T00:00:00.000Z"),
            "startPublishedDate": (str, "2023-01-01T00:00:00.000Z"),
            "endPublishedDate": (str, "2023-12-31T00:00:00.000Z"),
            "includeText": (list, ["large language model"]),
            "excludeText": (list, ["course"]),
            "highlightsQuery": (str, "Key advancements"),
            "summaryQuery": (str, "Main developments"),
        },
        response={
            # "requestId": "c6958155d5c89ffa0663b7c90c407396",
            "results": [
                {
                    "title": "A Comprehensive Overview of Large Language Models",
                    "url": "https://arxiv.org/pdf/2307.06435.pdf",
                    "publishedDate": "2023-11-16T01:36:32.547Z",
                    "author": "Humza  Naveed, University of Engineering and Technology (UET), Lahore, Pakistan",
                    "score": 0.4600165784358978,
                    "id": "https://arxiv.org/abs/2307.06435",
                    "image": "https://arxiv.org/pdf/2307.06435.pdf/page_1.png",
                    "favicon": "https://arxiv.org/favicon.ico",
                    "text": "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities...",
                    "highlights": ["Such requirements have limited their adoption..."],
                    "highlightScores": [0.4600165784358978],
                    "summary": "This overview paper on Large Language Models (LLMs) highlights key developments...",
                    "subpages": [
                        {
                            "id": "https://arxiv.org/abs/2303.17580",
                            "url": "https://arxiv.org/pdf/2303.17580.pdf",
                            "title": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
                            "author": "Yongliang  Shen, Microsoft Research Asia, Kaitao  Song, Microsoft Research Asia, Xu  Tan, Microsoft Research Asia, Dongsheng  Li, Microsoft Research Asia, Weiming  Lu, Microsoft Research Asia, Yueting  Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn, Zhejiang  University, Microsoft Research Asia, Microsoft  Research, Microsoft Research Asia",
                            "publishedDate": "2023-11-16T01:36:20.486Z",
                            "text": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face Date Published: 2023-05-25 Authors: Yongliang Shen, Microsoft Research Asia Kaitao Song, Microsoft Research Asia Xu Tan, Microsoft Research Asia Dongsheng Li, Microsoft Research Asia Weiming Lu, Microsoft Research Asia Yueting Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn Zhejiang University, Microsoft Research Asia Microsoft Research, Microsoft Research Asia Abstract Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower t",
                            "summary": "HuggingGPT is a framework using ChatGPT as a central controller to orchestrate various AI models from Hugging Face to solve complex tasks. ChatGPT plans the task, selects appropriate models based on their descriptions, executes subtasks, and summarizes the results. This approach addresses limitations of LLMs by allowing them to handle multimodal data (vision, speech) and coordinate multiple models for complex tasks, paving the way for more advanced AI systems.",
                            "highlights": [
                                "2) Recently, some researchers started to investigate the integration of using tools or models in LLMs  ."
                            ],
                            "highlightScores": [0.32679107785224915],
                        }
                    ],
                    "extras": {"links": []},
                }
            ],
            "costDollars": {
                "total": 0.005,
                "breakDown": [
                    {
                        "search": 0.005,
                        "contents": 0,
                        "breakdown": {
                            "keywordSearch": 0,
                            "neuralSearch": 0.005,
                            "contentText": 0,
                            "contentHighlight": 0,
                            "contentSummary": 0,
                        },
                    }
                ],
                "perRequestPrices": {
                    "neuralSearch_1_25_results": 0.005,
                    "neuralSearch_26_100_results": 0.025,
                    "neuralSearch_100_plus_results": 1,
                    "keywordSearch_1_100_results": 0.0025,
                    "keywordSearch_100_plus_results": 3,
                },
                "perPagePrices": {
                    "contentText": 0.001,
                    "contentHighlight": 0.001,
                    "contentSummary": 0.001,
                },
            },
        },
    )
    def find_similar(self, params: dict) -> dict:
        """
        Parameters:
            - url: The url for which you would like to find similar links.
                - string, required
                - Example: "https://arxiv.org/abs/2307.06435"
            - numResults: Number of results to return (up to thousands of results available for custom plans)
                - integer, optional, default:10
                - Required range: x <= 100
                - Example: 10
            - includeDomains: List of domains to include in the search. If specified, results will only come from these domains.
                - string[], optional
                - Example: ["arxiv.org", "paperswithcode.com"]
            - excludeDomains: List of domains to exclude from search results. If specified, no results will be returned from these domains.
                - string[], optional
            - startCrawlDate: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled after this date. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-01-01T00:00:00.000Z"
            - endCrawlDate: Crawl date refers to the date that Exa discovered a link. Results will include links that were crawled before this date. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-12-31T00:00:00.000Z"
            - startPublishedDate: Only links with a published date after this will be returned. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-01-01T00:00:00.000Z"
            - endPublishedDate: Only links with a published date before this will be returned. Must be specified in ISO 8601 format.
                - string, optional
                - Example: "2023-12-31T00:00:00.000Z"
            - includeText: List of strings that must be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.
                - string[], optional
                - Example: ["large language model"]
            - excludeText: List of strings that must not be present in webpage text of results. Currently, only 1 string is supported, of up to 5 words.
                - string[], optional
                - Example: ["course"]
            - highlightsQuery: Custom query to direct the LLM's selection of highlights.
                - string, optional
                - Example: "Key advancements"
            - summaryQuery: Custom query for the LLM-generated summary.
                - string, optional
                - Example: "Main developments"
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="contents",
        name="Get contents",
        description="Get the full page contents, summaries, and metadata for a list of URLs. Returns instant results from our cache, with automatic live crawling as fallback for uncached pages.",
        params={
            "urls*": (list, ["https://arxiv.org/abs/2307.06435"]),
            "highlightsQuery": (str, "Key advancements"),
            "summaryQuery": (str, "Main developments"),
        },
        response={
            # "requestId": "e492118ccdedcba5088bfc4357a8a125",
            "results": [
                {
                    "title": "A Comprehensive Overview of Large Language Models",
                    "url": "https://arxiv.org/pdf/2307.06435.pdf",
                    "publishedDate": "2023-11-16T01:36:32.547Z",
                    "author": "Humza  Naveed, University of Engineering and Technology (UET), Lahore, Pakistan",
                    "score": 0.4600165784358978,
                    "id": "https://arxiv.org/abs/2307.06435",
                    "image": "https://arxiv.org/pdf/2307.06435.pdf/page_1.png",
                    "favicon": "https://arxiv.org/favicon.ico",
                    "text": "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities...",
                    "highlights": ["Such requirements have limited their adoption..."],
                    "highlightScores": [0.4600165784358978],
                    "summary": "This overview paper on Large Language Models (LLMs) highlights key developments...",
                    "subpages": [
                        {
                            "id": "https://arxiv.org/abs/2303.17580",
                            "url": "https://arxiv.org/pdf/2303.17580.pdf",
                            "title": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
                            "author": "Yongliang  Shen, Microsoft Research Asia, Kaitao  Song, Microsoft Research Asia, Xu  Tan, Microsoft Research Asia, Dongsheng  Li, Microsoft Research Asia, Weiming  Lu, Microsoft Research Asia, Yueting  Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn, Zhejiang  University, Microsoft Research Asia, Microsoft  Research, Microsoft Research Asia",
                            "publishedDate": "2023-11-16T01:36:20.486Z",
                            "text": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face Date Published: 2023-05-25 Authors: Yongliang Shen, Microsoft Research Asia Kaitao Song, Microsoft Research Asia Xu Tan, Microsoft Research Asia Dongsheng Li, Microsoft Research Asia Weiming Lu, Microsoft Research Asia Yueting Zhuang, Microsoft Research Asia, yzhuang@zju.edu.cn Zhejiang University, Microsoft Research Asia Microsoft Research, Microsoft Research Asia Abstract Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower t",
                            "summary": "HuggingGPT is a framework using ChatGPT as a central controller to orchestrate various AI models from Hugging Face to solve complex tasks. ChatGPT plans the task, selects appropriate models based on their descriptions, executes subtasks, and summarizes the results. This approach addresses limitations of LLMs by allowing them to handle multimodal data (vision, speech) and coordinate multiple models for complex tasks, paving the way for more advanced AI systems.",
                            "highlights": [
                                "2) Recently, some researchers started to investigate the integration of using tools or models in LLMs  ."
                            ],
                            "highlightScores": [0.32679107785224915],
                        }
                    ],
                    "extras": {"links": []},
                }
            ],
            "costDollars": {
                "total": 0.005,
                "breakDown": [
                    {
                        "search": 0.005,
                        "contents": 0,
                        "breakdown": {
                            "keywordSearch": 0,
                            "neuralSearch": 0.005,
                            "contentText": 0,
                            "contentHighlight": 0,
                            "contentSummary": 0,
                        },
                    }
                ],
                "perRequestPrices": {
                    "neuralSearch_1_25_results": 0.005,
                    "neuralSearch_26_100_results": 0.025,
                    "neuralSearch_100_plus_results": 1,
                    "keywordSearch_1_100_results": 0.0025,
                    "keywordSearch_100_plus_results": 3,
                },
                "perPagePrices": {
                    "contentText": 0.001,
                    "contentHighlight": 0.001,
                    "contentSummary": 0.001,
                },
            },
        },
    )
    def contents(self, params: dict) -> dict:
        """
        Parameters:
            - urls: Array of URLs to crawl (backwards compatible with 'ids' parameter).
                - string[], required
                - Example: ["https://arxiv.org/pdf/2307.06435"]
            - highlightsQuery: Custom query to direct the LLM's selection of highlights.
                - string, optional
                - Example: "Key advancements"
            - summaryQuery: Custom query for the LLM-generated summary.
                - string, optional
                - Example: "Main developments"
        """
        return params
