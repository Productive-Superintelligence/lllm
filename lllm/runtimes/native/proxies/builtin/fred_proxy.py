# FRED Economic Data Proxy
# https://fred.stlouisfed.org/docs/api/fred/


import datetime as dt
import os

from ... import utils as U
from ..base import BaseProxy, ProxyRegistrator

REALTIME_PERIODS = """
- Real-Time Periods

The real-time period marks when facts were true or when information was known until it changed. Economic data sources, releases, series, and observations are all assigned a real-time period. Sources, releases, and series can change their names, and observation data values can be revised.
On almost all URLs, the default real-time period is today. This can be thought of as FRED® mode- what information about the past is available today. ALFRED® users can change the real-time period to retrieve information that was known as of a past period of history.
The real-time period can be specified by setting the realtime_start and realtime_end variables. Variables realtime_start and realtime_end are optional YYYY-MM-DD formatted dates that default to today's date. The real-time period set by realtime_start and realtime_end is a (closed, closed) period. This means that the real-time period includes the dates or boundaries set by realtime_start and realtime_end.

- Real-time Period for the 1980s Decade

To set the real-time period for the decade of the 1980s, set realtime_start = '1980-01-01' and realtime_end = '1989-12-31'.
To set the real-time period to 1980-01-01 and later, set realtime_start to '1980-01-01' and leave realtime_end unset or set realtime_end to '9999-12-31'.
To set the real-time period to 1980-01-01 and earlier, set realtime_end to '1980-01-01' and leave realtime_start unset or set realtime_start to '1776-07-04'.

- #growth_formulas: What formulas are used to calculate growth rates on the download data forms?

Note that because ALFRED uses levels and rounded data as published by the source, calculations of percentage changes and/or growth rates in some series may not be identical to those in the original releases.
The following formulas are used:
    - Change: x(t) - x(t-1)
    - Change from Year Ago: x(t) - x(t-n_obs_per_yr)
    - Percent Change: ((x(t)/x(t-1)) - 1) * 100
    - Percent Change from Year Ago: ((x(t)/x(t-n_obs_per_yr)) - 1) * 100
    - Compounded Annual Rate of Change: (((x(t)/x(t-1)) ** (n_obs_per_yr)) - 1) * 100
    - Continuously Compounded Rate of Change: (ln(x(t)) - ln(x(t-1))) * 100
    - Continuously Compounded Annual Rate of Change: ((ln(x(t)) - ln(x(t-1))) * 100) * n_obs_per_yr
    - Natural Log: ln(x(t))
Notes:
    - 'x(t)' is the value of series x at time period t.
    - 'n_obs_per_yr' is the number of observations per year. The number of observations per year differs by frequency:
        - Daily, 260 (no values on weekends)
        - Annual, 1
        - Monthly, 12
        - Quarterly, 4
        - Biweekly, 26
        - Weekly,52
    - 'ln' represents the natural logarithm.
    - '**' represents to the power of.
"""


@ProxyRegistrator(
    path="fred",
    name="Federal Reserve Economic Data",
    description=(
        "The FRED® API is a web service that allows developers to write programs and build applications that retrieve economic data from the FRED® and ALFRED® websites hosted by the Economic Research Division of the Federal Reserve Bank of St. Louis. Requests can be customized according to data source, release, category, series, and other preferences."
    ),
)
class FREDProxy(BaseProxy):
    """
    Federal Reserve Economic Data

    The Economic Research Division of the Federal Reserve Bank of St. Louis has enhanced the economic data services it provides by constructing an API (application programming interface), which allows users to create programs that retrieve data from our servers connected to the Internet.
    With our FRED® API, users may query our Federal Reserve Economic Data (FRED®) and Archival Federal Reserve Economic Data (ALFRED®) databases to retrieve the specific data desired (according to source, release, category and series among other preferences).
    """

    def __init__(self, cutoff_date: str = None, cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=cache, **kwargs)
        self.api_key_name = "api_key"
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred"
        self.enums = {}
        self.additional_docs = {"Real-Time Periods": REALTIME_PERIODS}

    def _call_api(
        self, url: str, params: dict, endpoint_info: dict, headers: dict
    ) -> dict:
        """
        Helper method to call the API using the requests library and remove specified keys.

        Args:
            url (str): The API endpoint URL.
            params (dict): Query parameters.
            remove_keys (list): List of keys to remove from the returned JSON response.

        Returns:
            dict: The filtered JSON response.
        """
        params["file_type"] = "json"

        if self.cutoff_date is not None:
            if "realtime_end" in params:
                realtime_end = dt.datetime.strptime(params["realtime_end"], "%Y-%m-%d")
                if realtime_end > self.cutoff_date:
                    realtime_start = dt.datetime.strptime(
                        params["realtime_start"], "%Y-%m-%d"
                    )
                    time_diff = realtime_end - realtime_start
                    params["realtime_end"] = self.cutoff_date.strftime("%Y-%m-%d")
                    params["realtime_start"] = (self.cutoff_date - time_diff).strftime(
                        "%Y-%m-%d"
                    )

            # set default realtime start and end
            if "realtime_start" not in params:
                params["realtime_start"] = self.cutoff_date.strftime("%Y-%m-%d")
            if "realtime_end" not in params:
                params["realtime_end"] = self.cutoff_date.strftime("%Y-%m-%d")

        response_json = U.call_api(url, params, headers, self.use_cache)
        return response_json

    ########################################
    ### Categories Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category",
        description="Get a category.",
        params={
            "category_id*": (int, 125),
        },
        response={
            "categories": [{"id": 125, "name": "Trade Balance", "parent_id": 13}]
        },
    )
    def category(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, default: 0 (root category)
        """
        return params

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category/children",
        description="Get the child categories for a specified parent category.",
        params={
            "category_id*": (int, 13),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
        },
        response={
            "categories": [
                {"id": 16, "name": "Exports", "parent_id": 13},
                {"id": 17, "name": "Imports", "parent_id": 13},
                {"id": 3000, "name": "Income Payments & Receipts", "parent_id": 13},
                {"id": 125, "name": "Trade Balance", "parent_id": 13},
                {"id": 127, "name": "U.S. International Finance", "parent_id": 13},
            ]
        },
    )
    def category_children(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, default: 0 (root category)
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        """
        return params

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category/related",
        description="Get the related categories for a category. A related category is a one-way relation between 2 categories that is not part of a parent-child category hierarchy. Most categories do not have related categories.",
        params={
            "category_id*": (int, 32073),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
        },
        response={
            "categories": [
                {"id": 149, "name": "Arkansas", "parent_id": 27281},
                {"id": 150, "name": "Illinois", "parent_id": 27281},
                {"id": 151, "name": "Indiana", "parent_id": 27281},
                {"id": 152, "name": "Kentucky", "parent_id": 27281},
                {"id": 153, "name": "Mississippi", "parent_id": 27281},
                {"id": 154, "name": "Missouri", "parent_id": 27281},
                {"id": 193, "name": "Tennessee", "parent_id": 27281},
            ]
        },
    )
    def category_related(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, default: 0 (root category)
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        """
        return params

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category/series",
        description="Get the series for a category.",
        params={
            "category_id*": (int, 125),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_id"),
            "sort_order": (str, "asc"),
            "filter_variable": (str, ""),
            "filter_value": (str, ""),
            "tag_names": (str, ""),
            "exclude_tag_names": (str, ""),
        },
        response={
            "realtime_start": "2017-08-01",
            "realtime_end": "2017-08-01",
            "order_by": "series_id",
            "sort_order": "asc",
            "count": 45,
            "offset": 0,
            "limit": 1000,
            "seriess": [
                {
                    "id": "BOPBCA",
                    "realtime_start": "2017-08-01",
                    "realtime_end": "2017-08-01",
                    "title": "Balance on Current Account (DISCONTINUED)",
                    "observation_start": "1960-01-01",
                    "observation_end": "2014-01-01",
                    "frequency": "Quarterly",
                    "frequency_short": "Q",
                    "units": "Billions of Dollars",
                    "units_short": "Bil. of $",
                    "seasonal_adjustment": "Seasonally Adjusted",
                    "seasonal_adjustment_short": "SA",
                    "last_updated": "2014-06-18 08:41:28-05",
                    "popularity": 32,
                    "group_popularity": 34,
                    "notes": "This series has been discontinued as a result of the comprehensive restructuring of the international economic accounts (http://www.bea.gov/international/modern.htm). For a crosswalk of the old and new series in FRED see: http://research.stlouisfed.org/CompRevisionReleaseID49.xlsx.",
                },
            ],
        },
    )
    def category_series(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_id', 'title', 'units', 'frequency', 'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end', 'popularity', 'group_popularity'.
            - optional, default: series_id
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        - filter_variable: The attribute to filter results by.
            - On of the following strings: 'frequency', 'units', 'seasonal_adjustment'.
            - optional, no filter by default
        - filter_value: The value of the filter_variable attribute to filter results by.
            - String, optional, no filter by default
        - tag_names: A semicolon delimited list of tag names that series match all of.
            - String, optional, no filtering by tags by default
            - Example value: 'income;bea'. Filter results to series having both tags 'income' and 'bea'.
            - See the related request tags.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - String, optional, no filtering by tags by default.
            - Example value: 'discontinued;annual'. Filter results to series having neither tag 'discontinued' nor tag 'annual'.
            - Parameter exclude_tag_names requires that parameter tag_names also be set to limit the number of matching series.
        """
        return params

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category/tags",
        description="Get the FRED tags for a category. Optionally, filter results by tag name, tag group, or search. Series are assigned tags and categories. Indirectly through series, it is possible to get the tags for a category. No tags exist for a category that does not have series. See the related request category/related_tags.",
        params={
            "category_id*": (int, 125),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "tag_names": (str, ""),
            "tag_group_id": (str, ""),
            "search_text": (str, ""),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "asc"),
        },
        response={
            "realtime_start": "2013-08-13",
            "realtime_end": "2013-08-13",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 21,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "bea",
                    "group_id": "src",
                    "notes": "U.S. Department of Commerce: Bureau of Economic Analysis",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 87,
                    "series_count": 24,
                },
            ],
        },
    )
    def category_tags(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names to only include in the response. See the related request category/related_tags.
            - String, optional, no filtering by tag names by default
            - Example value: 'trade;goods'. This value filters results to only include tags 'trade' and 'goods'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
        - search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    @BaseProxy.endpoint(
        category="Categories",
        endpoint="category/related_tags",
        description=(
            "Get the related FRED tags for one or more FRED tags within a category. Optionally, filter results by tag group or search. "
            "FRED tags are attributes assigned to series. For this request, related FRED tags are the tags assigned to series that match all tags in the tag_names parameter, no tags in the exclude_tag_names parameter, and the category set by the category_id parameter. See the related request category/tags. "
            "Series are assigned tags and categories. Indirectly through series, it is possible to get the tags for a category. No tags exist for a category that does not have series."
        ),
        params={
            "category_id*": (int, 125),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "tag_names*": (str, "services;quarterly"),
            "exclude_tag_names": (str, ""),
            "tag_group_id": (str, ""),
            "search_text": (str, ""),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "asc"),
        },
        response={
            "realtime_start": "2013-08-13",
            "realtime_end": "2013-08-13",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 7,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "balance",
                    "group_id": "gen",
                    "notes": "",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 65,
                    "series_count": 4,
                },
            ],
        },
    )
    def category_related_tags(self, params: dict) -> dict:
        """
        Parameters
        - category_id: The id for a category.
            - integer, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names that series match all of. See the related request category/tags.
            - String, required, no default value.
            - Example value: 'services;quarterly'. Find the related tags for series having both tags 'services' and 'quarterly'.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - String, optional, no default value.
            - Example value: 'goods;sa'. Find the related tags for series having neither tag 'goods' nor tag 'sa'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
        - search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    ########################################
    ### Series Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series",
        description="Get an economic data series.",
        params={
            "series_id*": (str, "GNPCA"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "seriess": [
                {
                    "id": "GNPCA",
                    "realtime_start": "2013-08-14",
                    "realtime_end": "2013-08-14",
                    "title": "Real Gross National Product",
                    "observation_start": "1929-01-01",
                    "observation_end": "2012-01-01",
                    "frequency": "Annual",
                    "frequency_short": "A",
                    "units": "Billions of Chained 2009 Dollars",
                    "units_short": "Bil. of Chn. 2009 $",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "seasonal_adjustment_short": "NSA",
                    "last_updated": "2013-07-31 09:26:16-05",
                    "popularity": 39,
                    "notes": "BEA Account Code: A001RX1",
                }
            ],
        },
    )
    def series(self, params: dict) -> dict:
        """
        Parameters
        - series_id: The id for a series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        """
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/categories",
        description="Get the categories for an economic data series.",
        params={
            "series_id*": (str, "EXJPUS"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
        },
        response={
            "categories": [
                {"id": 95, "name": "Monthly Rates", "parent_id": 15},
                {"id": 275, "name": "Japan", "parent_id": 158},
            ]
        },
    )
    def series_categories(self, params: dict) -> dict:
        """
        Parameters
        - series_id: The id for a series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        """
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/observations",
        description="Get the observations for an economic data series.",
        params={
            "series_id*": (str, "EXJPUS"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "limit": (int, 100000),
            "offset": (int, 0),
            "sort_order": (str, "asc"),
            "observation_start": (str, "1776-07-04"),
            "observation_end": (str, "9999-12-31"),
            "units": (str, "lin"),
            "frequency": (str, ""),
            "aggregation_method": (str, "avg"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "observation_start": "1776-07-04",
            "observation_end": "9999-12-31",
            "units": "lin",
            "output_type": 1,
            "file_type": "json",
            "order_by": "observation_date",
            "sort_order": "asc",
            "count": 84,
            "offset": 0,
            "limit": 100000,
            "observations": [
                {
                    "realtime_start": "2013-08-14",
                    "realtime_end": "2013-08-14",
                    "date": "1929-01-01",
                    "value": "1065.9",
                },
            ],
        },
    )
    def series_observations(self, params: dict) -> dict:
        """
        Parameters
        - series_id: The id for a series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        - observation_start: The start of the observation period.
            - YYYY-MM-DD formatted string, optional, default: 1776-07-04 (earliest available)
        - observation_end: The end of the observation period.
            - YYYY-MM-DD formatted string, optional, default: 9999-12-31 (latest available)
        - units: A key that indicates a data value transformation.
            - string, optional, default: lin (No transformation)
            - One of the following values: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'
                - lin = Levels (No transformation)
                - chg = Change
                - ch1 = Change from Year Ago
                - pch = Percent Change
                - pc1 = Percent Change from Year Ago
                - pca = Compounded Annual Rate of Change
                - cch = Continuously Compounded Rate of Change
                - cca = Continuously Compounded Annual Rate of Change
                - log = Natural Log
            - For unit transformation formulas, see #growth_formulas
        - frequency: An optional parameter that indicates a lower frequency to aggregate values to. The FRED frequency aggregation feature converts higher frequency data series into lower frequency data series (e.g. converts a monthly data series into an annual data series). In FRED, the highest frequency data is daily, and the lowest frequency data is annual. There are 3 aggregation methods available- average, sum, and end of period. See the aggregation_method parameter.
            - string, optional, default: no value for no frequency aggregation
            - One of the following values: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem'
                - Frequencies without period descriptions:
                    - d = Daily
                    - w = Weekly
                    - bw = Biweekly
                    - m = Monthly
                    - q = Quarterly
                    - sa = Semiannual
                    - a = Annual
                - Frequencies with period descriptions:
                    - wef = Weekly, Ending Friday
                    - weth = Weekly, Ending Thursday
                    - wew = Weekly, Ending Wednesday
                    - wetu = Weekly, Ending Tuesday
                    - wem = Weekly, Ending Monday
                    - wesu = Weekly, Ending Sunday
                    - wesa = Weekly, Ending Saturday
                    - bwew = Biweekly, Ending Wednesday
                    - bwem = Biweekly, Ending Monday
            - Note that an error will be returned if a frequency is specified that is higher than the native frequency of the series. For instance if a series has the native frequency 'Monthly' (as returned by the series request), it is not possible to aggregate the series to the higher 'Daily' frequency using the frequency parameter value 'd'.
            - No frequency aggregation will occur if the frequency specified by the frequency parameter matches the native frequency of the series. For instance if the value of the frequency parameter is 'm' and the native frequency of the series is 'Monthly' (as returned by the series request), observations will be returned, but they will not be aggregated to a lower frequency.
            - For most cases, it will be sufficient to specify a lower frequency without a period description (e.g. 'd', 'w', 'bw', 'm', 'q', 'sa', 'a') as opposed to frequencies with period descriptions (e.g. 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem') which only exist for the weekly and biweekly frequencies.
                - The weekly and biweekly frequencies with periods exist to offer more options and override the default periods implied by values 'w' and 'bw'.
                - The value 'w' defaults to frequency and period 'Weekly, Ending Friday' when aggregating daily series.
                - The value 'bw' defaults to frequency and period 'Biweekly, Ending Wednesday' when aggregating daily and weekly series.
                - Consider the difference between values 'w' for 'Weekly' and 'wef' for 'Weekly, Ending Friday'. When aggregating observations from daily to weekly, the value 'w' defaults to frequency and period 'Weekly, Ending Friday' which is the same as 'wef'. Here, the difference is that the period 'Ending Friday' is implicit for value 'w' but explicit for value 'wef'. However, if a series has native frequency 'Weekly, Ending Monday', an error will be returned for value 'wef' but not value 'w'.
            - Note that frequency aggregation is currently only available for file_type equal to xml or json due to time constraints.
            - Read the 'Frequency Aggregation' section of the FRED FAQs for implementation details.
        - aggregation_method: A key that indicates the aggregation method used for frequency aggregation. This parameter has no affect if the frequency parameter is not set.
            - string, optional, default: avg
            - One of the following values: 'avg', 'sum', 'eop'
                - avg = Average
                - sum = Sum
                - eop = End of Period
        """
        params["output_type"] = 1
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/search",
        description="Get economic data series that match search text.",
        params={
            "search_text*": (str, "monetary+service+index"),
            "search_type": (str, "full_text"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "search_rank"),
            "sort_order": (str, "desc"),
            "filter_variable": (str, ""),
            "filter_value": (str, ""),
            "tag_names": (str, ""),
            "exclude_tag_names": (str, ""),
        },
        response={
            "realtime_start": "2017-08-01",
            "realtime_end": "2017-08-01",
            "order_by": "search_rank",
            "sort_order": "desc",
            "count": 32,
            "offset": 0,
            "limit": 1000,
            "seriess": [
                {
                    "id": "MSIM2",
                    "realtime_start": "2017-08-01",
                    "realtime_end": "2017-08-01",
                    "title": "Monetary Services Index: M2 (preferred)",
                    "observation_start": "1967-01-01",
                    "observation_end": "2013-12-01",
                    "frequency": "Monthly",
                    "frequency_short": "M",
                    "units": "Billions of Dollars",
                    "units_short": "Bil. of $",
                    "seasonal_adjustment": "Seasonally Adjusted",
                    "seasonal_adjustment_short": "SA",
                    "last_updated": "2014-01-17 07:16:44-06",
                    "popularity": 34,
                    "group_popularity": 33,
                    "notes": "The MSI measure the flow of monetary services received each period by households and firms from their holdings of monetary assets (levels of the indexes are sometimes referred to as Divisia monetary aggregates).\nPreferred benchmark rate equals 100 basis points plus the largest rate in the set of rates.\nAlternative benchmark rate equals the larger of the preferred benchmark rate and the Baa corporate bond yield.\nMore information about the new MSI can be found at\nhttp://research.stlouisfed.org/msi/index.html.",
                },
                {
                    "id": "MSIM1P",
                    "realtime_start": "2017-08-01",
                    "realtime_end": "2017-08-01",
                    "title": "Monetary Services Index: M1 (preferred)",
                    "observation_start": "1967-01-01",
                    "observation_end": "2013-12-01",
                    "frequency": "Monthly",
                    "frequency_short": "M",
                    "units": "Billions of Dollars",
                    "units_short": "Bil. of $",
                    "seasonal_adjustment": "Seasonally Adjusted",
                    "seasonal_adjustment_short": "SA",
                    "last_updated": "2014-01-17 07:16:45-06",
                    "popularity": 26,
                    "group_popularity": 26,
                    "notes": "The MSI measure the flow of monetary services received each period by households and firms from their holdings of monetary assets (levels of the indexes are sometimes referred to as Divisia monetary aggregates).",
                },
            ],
        },
    )
    def series_search(self, params: dict) -> dict:
        """
        Parameters
        - search_text: The words to match against economic data series.
            - string, required
        - search_type: Determines the type of search to perform.
            - One of the following strings: 'full_text', 'series_id'.
            - 'full_text' searches series attributes title, units, frequency, and tags by parsing words into stems. This makes it possible for searches like 'Industry' to match series containing related words such as 'Industries'. Of course, you can search for multiple words like 'money' and 'stock'. Remember to url encode spaces (e.g. 'money%20stock').
            - 'series_id' performs a substring search on series IDs. Searching for 'ex' will find series containing 'ex' anywhere in a series ID. '*' can be used to anchor searches and match 0 or more of any character. Searching for 'ex*' will find series containing 'ex' at the beginning of a series ID. Searching for '*ex' will find series containing 'ex' at the end of a series ID. It's also possible to put an '*' in the middle of a string. 'm*sl' finds any series starting with 'm' and ending with 'sl'.
            - optional, default: full_text.
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'search_rank', 'series_id', 'title', 'units', 'frequency', 'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end', 'popularity', 'group_popularity'.
            - optional, default: If the value of search_type is 'full_text' then the default value of order_by is 'search_rank'. If the value of search_type is 'series_id' then the default value of order_by is 'series_id'.
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: If order_by is equal to 'search_rank' or 'popularity', then the default value of sort_order is 'desc'. Otherwise, the default sort order is 'asc'.
        - filter_variable: The attribute to filter results by.
            - One of the following strings: 'frequency', 'units', 'seasonal_adjustment'.
            - optional, no filter by default
        - filter_value: The value of the filter_variable attribute to filter results by.
            - string, optional, no filter by default
        - tag_names: A semicolon delimited list of tag names that series match all of.
            - string, optional, no filtering by tags by default
            - Example value: 'usa;m2'. Filter results to series having both tags 'usa' and 'm2'.
            - See the related request tags.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - string, optional, no filtering by tags by default
            - Example value: 'discontinued;m1'. Filter results to series having neither tag 'discontinued' nor tag 'm1'.
            - Parameter exclude_tag_names requires that parameter tag_names also be set to limit the number of matching series.
        """
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/search/tags",
        description="Get the FRED tags for a series search. Optionally, filter results by tag name, tag group, or tag search. See the related request series/search/related_tags.",
        params={
            "series_search_text*": (str, "monetary service index"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "tag_names": (str, ""),
            "tag_group_id": (str, ""),
            "tag_search_text": (str, ""),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "desc"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 18,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "academic data",
                    "group_id": "gen",
                    "notes": "Time series data created mainly by academia to address growing demand in understanding specific concerns in the economy that are not well modeled by ordinary statistical agencies.",
                    "created": "2012-08-29 10:22:19-05",
                    "popularity": 62,
                    "series_count": 25,
                },
                {
                    "name": "anderson & jones",
                    "group_id": "src",
                    "notes": "Richard Anderson and Barry Jones",
                    "created": "2013-06-21 10:22:49-05",
                    "popularity": 46,
                    "series_count": 25,
                },
            ],
        },
    )
    def series_search_tags(self, params: dict) -> dict:
        """
        Parameters
        - series_search_text: The words to match against economic data series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names to only include in the response. See the related request series/search/related_tags.
            - String, optional, no filtering by tag names by default
            - Example value: 'm1;m2'. This value filters results to only include tags 'm1' and 'm2'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
        - tag_search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/search/related_tags",
        description=(
            "Get the related FRED tags for one or more FRED tags matching a series search. Optionally, filter results by tag group or tag search. "
            "FRED tags are attributes assigned to series. For this request, related FRED tags are the tags assigned to series that match all tags in the tag_names parameter, no tags in the exclude_tag_names parameter, and the search words set by the series_search_text parameter. See the related request series/search/tags.",
        ),
        params={
            "series_search_text*": (str, "mortgage rate"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "tag_names*": (str, "30-year"),
            "exclude_tag_names": (str, ""),
            "tag_group_id": (str, ""),
            "tag_search_text": (str, ""),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "desc"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 10,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "conventional",
                    "group_id": "gen",
                    "notes": "",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 63,
                    "series_count": 3,
                },
                {
                    "name": "h15",
                    "group_id": "rls",
                    "notes": "H.15 Selected Interest Rates",
                    "created": "2012-08-16 15:21:17-05",
                    "popularity": 84,
                    "series_count": 3,
                },
            ],
        },
    )
    def series_search_related_tags(self, params: dict) -> dict:
        """
        Parameters
        - series_search_text: The words to match against economic data series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names that series match all of. See the related request series/search/tags.
            - String, optional, no filtering by tag names by default
            - Example value: '30-year;frb'. Find the related tags for series having both tags '30-year' and 'frb'.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - String, optional, no default value.
            - Example value: 'discontinued;monthly'. Find the related tags for series having neither tag 'discontinued' nor tag 'monthly'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
        - tag_search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    @BaseProxy.endpoint(
        category="Series",
        endpoint="series/tags",
        description="Get the FRED tags for a series.",
        params={
            "series_id*": (str, "STLFSI"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "order_by": (str, "series_count"),
            "sort_order": (str, "desc"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 8,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "nation",
                    "group_id": "geot",
                    "notes": "Country Level",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 100,
                    "series_count": 105200,
                },
                {
                    "name": "nsa",
                    "group_id": "seas",
                    "notes": "Not seasonally adjusted",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 96,
                    "series_count": 100468,
                },
            ],
        },
    )
    def series_tags(self, params: dict) -> dict:
        """
        Parameters
        - series_id: The id for a series.
            - string, required
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    ########################################
    ### Tags Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Tags",
        endpoint="tags",
        description="Get FRED tags. Optionally, filter results by tag name, tag group, or search. FRED tags are attributes assigned to series. See the related request related_tags.",
        params={
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "tag_names": (str, ""),
            "tag_group_id": (str, ""),
            "search_text": (str, ""),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "desc"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 4794,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "nation",
                    "group_id": "geot",
                    "notes": "Country Level",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 100,
                    "series_count": 105200,
                },
                {
                    "name": "nsa",
                    "group_id": "seas",
                    "notes": "Not seasonally adjusted",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 96,
                    "series_count": 100468,
                },
            ],
        },
    )
    def tags(self, params: dict) -> dict:
        """
        Parameters
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names to only include in the response. See the related request related_tags.
            - String, optional, no filtering by tag names by default
            - Example value: 'gdp;oecd'. This value filters results to only include tags 'gdp' and 'oecd'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src', 'cc'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
                - cc = Citation & Copyright
        - search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    @BaseProxy.endpoint(
        category="Tags",
        endpoint="related_tags",
        description=(
            "Get the related FRED tags for one or more FRED tags. Optionally, filter results by tag group or search.",
            "FRED tags are attributes assigned to series. Related FRED tags are the tags assigned to series that match all tags in the tag_names parameter and no tags in the exclude_tag_names parameter. See the related request tags.",
        ),
        params={
            "tag_names*": (str, "monetary aggregates;weekly"),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "exclude_tag_names": (str, None),
            "tag_group_id": (str, None),
            "search_text": (str, None),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_count"),
            "sort_order": (str, "desc"),
        },
        response={
            "realtime_start": "2013-08-14",
            "realtime_end": "2013-08-14",
            "order_by": "series_count",
            "sort_order": "desc",
            "count": 13,
            "offset": 0,
            "limit": 1000,
            "tags": [
                {
                    "name": "nation",
                    "group_id": "geot",
                    "notes": "Country Level",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 100,
                    "series_count": 12,
                },
                {
                    "name": "usa",
                    "group_id": "geo",
                    "notes": "United States of America",
                    "created": "2012-02-27 10:18:19-06",
                    "popularity": 100,
                    "series_count": 12,
                },
            ],
        },
    )
    def related_tags(self, params: dict) -> dict:
        """
        Parameters
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - tag_names: A semicolon delimited list of tag names that series match all of. See the related request tags.
            - String, required, no default value.
            - Example value: 'monetary+aggregates;weekly'. Find the related tags for series having both tags 'monetary aggregates' and 'weekly'. The '+' in 'monetary+aggregates;weekly' is an URL encoded space character.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - String, optional, no default value.
            - Example value: 'discontinued;currency'. Find the related tags for series having neither tag 'discontinued' nor tag 'currency'.
        - tag_group_id: A tag group id to filter tags by type.
            - String, optional, no filtering by tag group by default.
            - One of the following: 'freq', 'gen', 'geo', 'geot', 'rls', 'seas', 'src'.
                - freq = Frequency
                - gen = General or Concept
                - geo = Geography
                - geot = Geography Type
                - rls = Release
                - seas = Seasonal Adjustment
                - src = Source
        - search_text: The words to find matching tags with.
            - String, optional, no filtering by search words by default.
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_count', 'popularity', 'created', 'name', 'group_id'.
            - optional, default: series_count
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params

    @BaseProxy.endpoint(
        category="Tags",
        endpoint="tags/series",
        description="Get the series matching all tags in the tag_names parameter and no tags in the exclude_tag_names parameter.",
        params={
            "tag_names*": (str, "slovenia;food;oecd"),
            "exclude_tag_names": (str, ""),
            "realtime_start": (str, "2013-08-14"),
            "realtime_end": (str, "2013-08-14"),
            "limit": (int, 1000),
            "offset": (int, 0),
            "order_by": (str, "series_id"),
            "sort_order": (str, "asc"),
        },
        response={
            "realtime_start": "2017-08-01",
            "realtime_end": "2017-08-01",
            "order_by": "series_id",
            "sort_order": "asc",
            "count": 18,
            "offset": 0,
            "limit": 1000,
            "seriess": [
                {
                    "id": "CPGDFD02SIA657N",
                    "realtime_start": "2017-08-01",
                    "realtime_end": "2017-08-01",
                    "title": "Consumer Price Index: Total Food Excluding Restaurants for Slovenia\u00a9",
                    "observation_start": "1996-01-01",
                    "observation_end": "2016-01-01",
                    "frequency": "Annual",
                    "frequency_short": "A",
                    "units": "Growth Rate Previous Period",
                    "units_short": "Growth Rate Previous Period",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "seasonal_adjustment_short": "NSA",
                    "last_updated": "2017-04-20 00:48:35-05",
                    "popularity": 0,
                    "group_popularity": 0,
                    "notes": 'OECD descriptor ID: CPGDFD02\nOECD unit ID: GP\nOECD country ID: SVN\n\nAll OECD data should be cited as follows: OECD, "Main Economic Indicators - complete database", Main Economic Indicators (database),http://dx.doi.org/10.1787/data-00052-en (Accessed on date)\nCopyright, 2016, OECD. Reprinted with permission.',
                },
                {
                    "id": "CPGDFD02SIA659N",
                    "realtime_start": "2017-08-01",
                    "realtime_end": "2017-08-01",
                    "title": "Consumer Price Index: Total Food Excluding Restaurants for Slovenia\u00a9",
                    "observation_start": "1996-01-01",
                    "observation_end": "2016-01-01",
                    "frequency": "Annual",
                    "frequency_short": "A",
                    "units": "Growth Rate Same Period Previous Year",
                    "units_short": "Growth Rate Same Period Previous Yr.",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "seasonal_adjustment_short": "NSA",
                    "last_updated": "2017-04-20 00:48:35-05",
                    "popularity": 0,
                    "group_popularity": 0,
                    "notes": 'OECD descriptor ID: CPGDFD02\nOECD unit ID: GY\nOECD country ID: SVN\n\nAll OECD data should be cited as follows: OECD, "Main Economic Indicators - complete database", Main Economic Indicators (database),http://dx.doi.org/10.1787/data-00052-en (Accessed on date)\nCopyright, 2016, OECD. Reprinted with permission.',
                },
            ],
        },
    )
    def tags_series(self, params: dict) -> dict:
        """
        Parameters
        - tag_names: A semicolon delimited list of tag names that series match all of.
            - String, required, no default value.
            - Example value: 'slovenia;food'. Filter results to series having both tags 'slovenia' and 'food'.
            - See the related request tags.
        - exclude_tag_names: A semicolon delimited list of tag names that series match none of.
            - String, optional, no default value.
            - Example value: 'alchohol;quarterly'. Filter results to series having neither tag 'alchohol' nor tag 'quarterly'.
        - realtime_start: The start of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - realtime_end: The end of the real-time period. For more information, see Real-Time Periods.
            - YYYY-MM-DD formatted string, optional, default: today's date
        - limit: The maximum number of results to return.
            - integer between 1 and 1000, optional, default: 1000
        - offset: non-negative integer, optional, default: 0
        - order_by: Order results by values of the specified attribute.
            - One of the following strings: 'series_id', 'title', 'units', 'frequency', 'seasonal_adjustment', 'realtime_start', 'realtime_end', 'last_updated', 'observation_start', 'observation_end', 'popularity', 'group_popularity'.
            - optional, default: series_id
        - sort_order: Sort results is ascending or descending order for attribute values specified by order_by.
            - One of the following strings: 'asc', 'desc'.
            - optional, default: asc
        """
        return params
