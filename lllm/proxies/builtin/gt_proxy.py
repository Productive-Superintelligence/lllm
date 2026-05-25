# Google Trends Search API Proxy
# https://www.searchapi.io/docs/google-trends


import datetime as dt
import os

from ... import utils as U
from ..base import BaseProxy, ProxyRegistrator

FILE_PATH = os.path.dirname(
    os.path.abspath(__file__)
)  # /home/junyanc/analytical_engine/analytica/proxy/modules


@ProxyRegistrator(
    path="gt",
    name="Google Trends Search API",
    description=(
        "Google Trends API scrape real-time results from Google Trends. It also supports autocomplete, related queries, related topics, and geo locations."
    ),
)
class GTProxy(BaseProxy):
    """
    Google Trends Search API

    This API provides access to Google Trends Search API.
    """

    def __init__(self, cutoff_date: str = None, cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=cache, **kwargs)
        self.api_key_name = "api_key"
        self.api_key = os.getenv("SEARCH_API_KEY")
        self.base_url = "https://www.searchapi.io/api/v1/search"
        self.enums = {}

        self._google_trends_categories = U.load_json(
            U.pjoin(FILE_PATH, "google-trends-categories.json"), None
        )
        self._google_trends_geo = U.load_json(
            U.pjoin(FILE_PATH, "google-trends-geo.json"), None
        )

    def _call_api(
        self, url: str, params: dict, endpoint_info: dict, headers: dict
    ) -> dict:
        """
        Helper method to call the API using the requests library.
        """
        endpoint = endpoint_info["endpoint"]
        if endpoint == "google_trends_geo":
            return self._google_trends_geo
        elif endpoint == "google_trends_categories":
            return self._google_trends_categories
        else:
            url = self.base_url
        response_json = U.call_api(url, params, headers, self.use_cache)
        return response_json

    def _google_trends(self, params: dict):
        """
        Search for a keyword on Google Trends.

        Parameters:
            - q (required or optional): Search Query. Parameter defines the search query. It can be either required or optional based on the `data_type` parameter:
                - `data_type=TIMESERIES` - search query is required only if `cat=0` (default). It is limited to 5 search queries that should be seperated by `,`. For instance: `Java,JavaScript,Python`.
                - `data_type=GEO_MAP` - search query is required.
                - `data_type=RELATED_QUERIES` - search query is optional.
                - `data_type=RELATED_TOPICS` - search query is optional.
            - data_type (required): Data Type. Parameter defines the data type you wish to search for. You have several options to choose from:
                - `TIMESERIES` also known as Interest Over Time, used to return historical, indexed data for a given input.
                - `GEO_MAP` also known as Interest Over Region, used for geographical data.
                - `RELATED_QUERIES` corresponds to searches for queries that are related to the given input.
                - `RELATED_TOPICS` represents searches related to specific topics.
            - cat (optional): Categories. The parameter for category selection defaults to `0`, representing All Categories. This parameter determines the category to be used for the specified search.
                - Please refer to the complete list of supported Google Trends Categories by calling "google_trends_categories" endpoint for more details.
            - region (optional): Region. The parameter specifies the geographical region for your chosen search. It is usable only with `GEO_MAP` data_type. There are few options to consider:
                - `COUNTRY` - selects data from country searches. Usable only with `Worldwide` geo value.
                - `REGION` - selects data from a states or provinces.
                - `DMA` - selects data from metros.
                - `CITY` - selects data from cities.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months (default).
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - Please always use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019.
                - Note: `tz` parameter significantly influences the results.
        """
        params["engine"] = "google_trends"

        if self.cutoff_date is not None:
            if "time" not in params:
                params["time"] = "today 12-m"
            _time = params["time"]
            end_date = self.cutoff_date.strftime("%Y-%m-%d")
            if _time == "all":
                params["time"] = "2004-01-01 " + end_date
            elif "today" in _time:
                shift = _time.split(" ")[1]
                shift_days = {
                    "1-d": 1,
                    "7-d": 7,
                    "1-m": 30,
                    "3-m": 90,
                    "12-m": 365,
                    "5-y": 1825,
                }[shift]
                params["time"] = (
                    (self.cutoff_date - dt.timedelta(days=shift_days)).strftime(
                        "%Y-%m-%d"
                    )
                    + " "
                    + end_date
                )
            else:
                _from, _to = _time.split(" ")
                from_date = dt.datetime.strptime(_from, "%Y-%m-%d")
                to_date = dt.datetime.strptime(_to, "%Y-%m-%d")
                date_diff = (to_date - from_date).days
                start_date = self.cutoff_date - dt.timedelta(days=date_diff)
                params["time"] = start_date.strftime("%Y-%m-%d") + " " + end_date
        return params

    ########################################
    ### Google Trends Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_overtime",
        name="Google Trends Interest Over Time",
        description="Search for one or more keywords on Google Trends interest over time",
        params={
            "q*": (str, "keyword"),
            "geo": (str, "US"),
            "tz": (int, 420),
            "gprop": (str, ""),
            "time": (str, "today 12-m"),
        },
        response={
            "interest_over_time": {
                "averages": [
                    {"query": "Java", "value": 58},
                    {"query": "Python", "value": 85},
                ],
                "timeline_data": [
                    {
                        "date": "May 1 – 7, 2022",
                        "timestamp": "1651363200",
                        "values": [
                            {"query": "Java", "value": "59", "extracted_value": 59},
                            {"query": "Python", "value": "87", "extracted_value": 87},
                        ],
                    },
                ],
            }
        },
    )
    def google_trends_overtime(self, params: dict):
        """
        Parameters:
            - q (required): Search Query. Parameter defines the search query.
                - It is limited to 5 search queries that should be seperated by `,`. For instance: `Java,JavaScript,Python`.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months.
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - To select a custom date range, use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019. If you want to select a specific hourly range within the past week, use the format `yyyy-mm-ddThh`. For instance, `2025-03-23T21 2025-03-24T04` will retrieve data from 9PM on 2025-03-23, until 4AM on 2025-03-24.
                - Note: `tz` parameter significantly influences the results, and hourly range selections are limited to data from the previous week.
        """
        params["data_type"] = "TIMESERIES"
        params["cat"] = 0
        return self._google_trends(params)

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_by_category",
        name="Google Trends Interest Over Time by Category",
        description="Search for one or more keywords on Google Trends interest over time by category",
        params={
            "cat*": (int, 3),
            "q": (str, ""),
            "geo": (str, ""),
            "tz": (int, 420),
            "gprop": (str, ""),
            "time": (str, "today 12-m"),
        },
        response={
            "interest_over_time": {
                "timeline_data": [
                    {
                        "date": "Mar 24 – 30, 2024",
                        "timestamp": "1711238400",
                        "values": [{"value": "91", "extracted_value": 91}],
                    },
                ]
            }
        },
    )
    def google_trends_by_category(self, params: dict):
        """
        Parameters:
            - cat (required): Categories. This parameter determines the category to be used for the specified search.
                - Please refer to the complete list of supported Google Trends Categories by calling "google_trends_categories" endpoint for more details.
            - q (optional): Search Query. Parameter defines the search query.
                - It is limited to 5 search queries that should be seperated by `,`. For instance: `Java,JavaScript,Python`.
                - If not provided, the trend of the category will be returned.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months.
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - To select a custom date range, use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019. If you want to select a specific hourly range within the past week, use the format `yyyy-mm-ddThh`. For instance, `2025-03-23T21 2025-03-24T04` will retrieve data from 9PM on 2025-03-23, until 4AM on 2025-03-24.
                - Note: `tz` parameter significantly influences the results, and hourly range selections are limited to data from the previous week.
        """
        params["data_type"] = "TIMESERIES"
        return self._google_trends(params)

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_by_region",
        name="Google Trends Interest Over Region",
        description="Search for one or more keywords on Google Trends Interest Over Region, used for geographical data.",
        params={
            "q*": (str, "Java,Python,Ruby"),
            "region": (str, "COUNTRY"),
            "cat": (int, 0),
            "geo": (str, ""),
            "tz": (int, 420),
            "gprop": (str, ""),
            "time": (str, "today 12-m"),
        },
        response={
            "interest_by_region": [
                {
                    "geo": "US",
                    "name": "United States",
                    "values": [
                        {"query": "Java", "value": "20%", "extracted_value": 20},
                        {"query": "Python", "value": "40%", "extracted_value": 40},
                    ],
                },
            ]
        },
    )
    def google_trends_by_region(self, params: dict):
        """
        Parameters:
            - q (required): Search Query. Parameter defines the search query.
            - cat (optional): Categories. The parameter for category selection defaults to `0`, representing All Categories. This parameter determines the category to be used for the specified search.
                - Please refer to the complete list of supported Google Trends Categories by calling "google_trends_categories" endpoint for more details.
            - region (optional): Region. The parameter specifies the geographical region for your chosen search. There are few options to consider:
                - `COUNTRY` - selects data from country searches. Usable only with `Worldwide` geo value.
                - `REGION` - selects data from a states or provinces.
                - `DMA` - selects data from metros.
                - `CITY` - selects data from cities.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months.
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - To select a custom date range, use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019. If you want to select a specific hourly range within the past week, use the format `yyyy-mm-ddThh`. For instance, `2025-03-23T21 2025-03-24T04` will retrieve data from 9PM on 2025-03-23, until 4AM on 2025-03-24.
                - Note: `tz` parameter significantly influences the results, and hourly range selections are limited to data from the previous week.
        """
        params["data_type"] = "GEO_MAP"
        return self._google_trends(params)

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_related_queries",
        name="Google Trends Related Queries",
        description="Searches for queries that are related to the given input.",
        params={
            "q*": (str, "Java"),
            "cat": (int, 0),
            "geo": (str, "US"),
            "tz": (int, 420),
            "gprop": (str, ""),
            "time": (str, "today 12-m"),
        },
        response={
            "related_queries": {
                "top": [
                    {
                        "position": 1,
                        "query": "minecraft",
                        "values": "100",
                        "extracted_value": 100,
                        "link": "https://trends.google.com/trends/explore?q=minecraft&date=today+12-m",
                    },
                ],
                "rising": [
                    {
                        "position": 1,
                        "query": "tynker",
                        "values": "100",
                        "extracted_value": 100,
                        "link": "https://trends.google.com/trends/explore?q=tynker&date=today+12-m",
                    },
                ],
            }
        },
    )
    def google_trends_related_queries(self, params: dict):
        """
        Parameters:
            - q (required): Search Query. Parameter defines the search query.
            - cat (optional): Categories. The parameter for category selection defaults to `0`, representing All Categories. This parameter determines the category to be used for the specified search.
                - Please refer to the complete list of supported Google Trends Categories by calling "google_trends_categories" endpoint for more details.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months.
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - To select a custom date range, use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019. If you want to select a specific hourly range within the past week, use the format `yyyy-mm-ddThh`. For instance, `2025-03-23T21 2025-03-24T04` will retrieve data from 9PM on 2025-03-23, until 4AM on 2025-03-24.
                - Note: `tz` parameter significantly influences the results, and hourly range selections are limited to data from the previous week.
        """
        params["data_type"] = "RELATED_QUERIES"
        return self._google_trends(params)

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_related_topics",
        name="Google Trends Related Topics",
        description="Searches for topics that are related to the given input.",
        params={
            "q*": (str, "Java"),
            "cat": (int, 0),
            "geo": (str, "US"),
            "tz": (int, 420),
            "gprop": (str, ""),
            "time": (str, "today 12-m"),
        },
        response={
            "related_topics": {
                "top": [
                    {
                        "position": 1,
                        "id": "/m/05z1_",
                        "title": "Python",
                        "type": "Programming language",
                        "value": "100",
                        "extracted_value": 100,
                        "link": "https://trends.google.com/trends/explore?q=/m/05z1_&date=today+12-m",
                    },
                ],
                "rising": [
                    {
                        "position": 1,
                        "id": "/m/11ksffl5ml",
                        "title": "LeetCode",
                        "type": "Software industry company",
                        "value": "100",
                        "extracted_value": 100,
                        "link": "https://trends.google.com/trends/explore?q=/m/11ksffl5ml&date=today+12-m",
                    }
                ],
            }
        },
    )
    def google_trends_related_topics(self, params: dict):
        """
        Parameters:
            - q (required): Search Query. Parameter defines the search query.
            - cat (optional): Categories. The parameter for category selection defaults to `0`, representing All Categories. This parameter determines the category to be used for the specified search.
                - Please refer to the complete list of supported Google Trends Categories by calling "google_trends_categories" endpoint for more details.
            - geo (optional): Localization. The default value for the location parameter is set to `Worldwide`, which denotes a global scope for the search. This parameter specifies the geographical area for the query search. If it is not explicitly set, the search defaults to a worldwide range.
                - Check the full list of supported Google Trends geo locations by calling google_trends_geo endpoint for more details.
            - tz (optional): Timezone. Parameter defines timezone offset (the difference in hours and minutes between a particular time zone and UTC). Could be selected from `-1439` to `1439`. Default - `420`.
            - gprop (optional): Filter. This parameter can be customized according to different search types, with each corresponding to a distinct functionality. The options include:
                - `""` represents Web Search (Default value).
                - `images` represents Image Search.
                - `news` represents News Search.
                - `froogle` represents Google Shopping.
                - `youtube` represents YouTube Search.
            - time (optional): Filtering by time. The parameter determines the time range for the data retrieval. There are a few options:
                - `today 1-d` - data from the past day.
                - `today 7-d` - data from the past 7 days.
                - `today 1-m` - data from the past 30 days.
                - `today 3-m` - data from the past 90 days.
                - `today 12-m` - data from the past 12 months.
                - `today 5-y` - data from the past 5 years.
                - `all` - All available data since 2004.
                - To select a custom date range, use the format `yyyy-mm-dd`. For example, `2019-01-01 2019-12-31` will retrieve data for the entire year of 2019. If you want to select a specific hourly range within the past week, use the format `yyyy-mm-ddThh`. For instance, `2025-03-23T21 2025-03-24T04` will retrieve data from 9PM on 2025-03-23, until 4AM on 2025-03-24.
                - Note: `tz` parameter significantly influences the results, and hourly range selections are limited to data from the previous week.
        """
        params["data_type"] = "RELATED_TOPICS"
        return self._google_trends(params)

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_categories",
        name="Google Trends Categories",
        description="Returns a list of all available Google Trends categories.",
        params={},
        response=[
            {"category_id": "0", "category_description": "All categories"},
            {"category_id": "3", "category_description": "Arts & Entertainment"},
        ],
    )
    def google_trends_categories(self, params: dict):
        """
        Parameters:
            - None
        """
        return params

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_geo",
        name="Google Trends Geo",
        description="Returns a list of all available Google Trends geo locations.",
        params={},
        response=[
            {"geo_id": "", "geo_description": "Worldwide"},
            {"geo_id": "AF", "geo_description": "Afghanistan"},
        ],
    )
    def google_trends_geo(self, params: dict):
        """
        Parameters:
            - None
        """
        return params

    @BaseProxy.endpoint(
        category="Google Trends",
        endpoint="google_trends_autocomplete",
        name="Google Trends Autocomplete",
        description="Returns a list of Google Trends autocomplete suggestions.",
        params={
            "q*": (str, "Programming language"),
        },
        response={
            "suggestions": [
                {
                    "id": "/m/01t6b",
                    "title": "C",
                    "type": "Programming language",
                    "link": "https://trends.google.com/trends/explore?q=/m/01t6b",
                    "thumbnail": "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcT2oSmZNi5-GY7XES_71nsMGIDT3vGnKce3B9hzryxuCPCBIzWp",
                },
            ]
        },
    )
    def google_trends_autocomplete(self, params: dict):
        """
        Parameters:
            - q (required): Search Query.
        """
        params["engine"] = "google_trends_autocomplete"
        return params
