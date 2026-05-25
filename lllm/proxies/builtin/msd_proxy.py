# Main Street Data Proxy
# MSD API https://mainstreetdata.com/home 
# https://api.mainstreetdata.com/docs/#/


import os
from lllm.proxies.base import BaseProxy, ProxyRegistrator




@ProxyRegistrator(
    path='msd', 
    name='Main Street Data API', 
    description=(
        "The Main Street Data API is an invaluable resource. The team behind this innovative product has meticulously compiled over thousands of metrics related to 2,500 US companies, offering unparalleled insights into businesses beyond standard financial statements. These KPIs provide a direct measure of business quality, giving investors a significant advantage by offering rare, hard-to-source data."
    )
)
class MSDProxy(BaseProxy):
    """
    Main Street Data

    This API provides access to curated KPI data from MainStreet Data. The data includes detailed individual operating metrics, such as Tesla’s Supercharger Stations and Palantir’s Customer Count, as well as segmented metrics like Nvidia’s Operating Income, empowering users with granular and sector-specific insights.

    Some of the most popular metrics:
     - TSLA Vehicles Delivered by Model
     - TSLA Gross Margin by Segment
     - META Family Daily Active Users
     - NVDA Operating Income by Segment
     - SPOT Premium Users
     - PLTR Total Customers
     - AWS Performance Obligations
     - AAPL Global Active Devices
     - SOFI Contribution Profit by Segment
     - SQ Gross Payment Volume
     - UBER Take Rate by Segment
     - HOOD Assets Under Custody
     - HIMS Subscriptions
     - ABNB Gross Booking Value
     - SHOP Monthly Recurring Revenue
     - … 
    
    TSLA metrics example:
     - Vehicles Delivered by Model
     - Energy Storage Deployed
     - Supercharger Stations
     - Stores and Service Locations
     - Mobile Service Fleet
     - Vehicle Inventory Days of Supply
     - MSRP by Base Model
     - Revenue by Segment
     - Revenue by Geography
     - Per Vehicle Returns
     - Gross Profit by Segment
     - Gross Margin by Segment
     - Operating Expense Breakdown
    """
    def __init__(self, cutoff_date: str = None, cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=cache, **kwargs)
        self.api_key_name = "*x-api-key" # * means it's a header
        self.api_key = os.getenv("MSD_API_KEY")
        self.base_url = "https://api.mainstreetdata.com/api/v1"
        self.enums = {}


    ########################################
    ### Companies Endpoints
    ########################################

    @BaseProxy.endpoint(
        category='Companies',
        endpoint='companies/{ticker}',
        name='Company Data',
        description='Retrieve company data and its associated metrics.',
        params={
            "$ticker*": (str, "AAPL"), # $ means path parameter
            "freq": (str, "quarterly"),
            "metricName": (str, 'aapl_americas_msd'),
            "startDate": (str, '2023-01-01'),
            "endDate": (str, '2023-12-31'),
            "yoy": (bool, False),
            "percentRevenue": (bool, False),
        },
        response={
            "company": {
                "ticker": "AAPL",
                "name": "Apple",
                "totalMetrics": 1
            },
            "metrics": [
                {
                "name": "Americas Revenue",
                "type": "cumulative",
                "category": "Revenue by Geography",
                "values": [
                    {
                        "x": "2023-03-31T00:00:00.000Z",
                        "y": 37874000000,
                        "valueType": "CURRENCY"
                    },
                    {
                        "x": "2023-06-30T00:00:00.000Z",
                        "y": 35383000000,
                        "valueType": "CURRENCY"
                    },
                    {
                        "x": "2023-09-30T00:00:00.000Z",
                        "y": 40115000000,
                        "valueType": "CURRENCY"
                    },
                    {
                        "x": "2023-12-31T00:00:00.000Z",
                        "y": 50430000000,
                        "valueType": "CURRENCY"
                    }
                ]
                }
            ]
        }
    )
    def companies_ticker(self, params: dict):
        """
        Fetch the details of a specific company, including its metrics and data points, based on the provided ticker symbol.

        Parameters:
            - ticker: The stock ticker symbol of the company (e.g., AAPL, MSFT, SOFI). 
                - string, required.
            - freq: freq can take 1 of the following values (annual, quarterly, ttm) and defaults to quarterly. NOTE: There are some metrics that are only reported annually.
                - string, optional.
                - Available values : annual, quarterly, ttm
            - metricName: The name of the metric to retrieve. If not provided, all metrics will be returned. (e.g. aapl_americas_msd)
                - string, optional.
            - startDate: The start date for the data points (inclusive). (e.g. 2023-01-01)
                - string($date), optional.
            - endDate: The end date for the data points (inclusive). (e.g. 2023-12-31)
                - string($date), optional.
            - yoy: A flag to indicate if Year-over-Year (YoY) calculations should be included. If true, the response will include YoY percentage changes for the dataset.
                - boolean, optional.
            - percentRevenue: If true, the response will also include % of revenue of the datapoint.
                - boolean, optional.
        """
        return params
    


    @BaseProxy.endpoint(
        category='Companies',
        endpoint='companies',
        name='Company List',
        description='Retrieve all available companies.',
        params={},
        response=[
            "SOFI",
        ],
    )
    def companies_list(self, params: dict):
        """
        Fetch a list of all company tickers available for querying.

        Parameters:
        No parameters
        """
        return params


    @BaseProxy.endpoint(
        category='Companies',
        endpoint='companies/{ticker}/kpi',
        name='KPI',
        description='Retrieve all available metric names for a specific company.',
        params={
            "$ticker*": (str, "AAPL"), 
        },
        response=[
            {
                "label": "Global Active Devices",
                "columnName": "aapl_total_active_devices_msd",
                "onlyAnnual": True,
                "category": "Operating Metrics"
            },
            {
                "label": "iPhone",
                "columnName": "aapl_iphone_msd",
                "onlyAnnual": False,
                "category": "Revenue by Segment"
            },
            {
                "label": "Americas",
                "columnName": "aapl_americas_msd",
                "onlyAnnual": False,
                "category": "Revenue by Geography"
            }
        ],
    )
    def kpi(self, params: dict):
        """
        Fetch a list of all relevant information for the KPI metrics of a specific company based on the provided ticker symbol. This includes the metric label, column name, category, and whether the metric is only reported annually.

        Parameters:
            - ticker: The stock ticker symbol of the company (e.g., AAPL, MSFT). 
                - string, required.

        Errors:
            - 404: Company not found.
                - Response:
                    {
                        "error": "Company with ticker 'SOFI' not found"
                    }
        """
        return params

    @BaseProxy.endpoint(
        category='Companies',
        endpoint='companies', 
        method='POST',
        name='Multiple Companies Data',
        description='Retrieve company data for multiple tickers and their associated metrics.',
        params={
            "#tickers*": (list, ["AAPL", "MSFT"]), # # means post request body item
            "metrics": (list, ["aapl_iphone_msd", "msft_total_active_devices_msd"]),
            "startDate": (str, "2023-01-01"),
            "endDate": (str, "2023-12-31"),
        },
        response={
            "companies": [
                {
                    "company": {
                        "ticker": "AAPL",
                        "name": "Apple Inc."
                    },
                    "metrics": [
                        {
                            "name": "Research and Development",
                            "leftSymbol": "$",
                            "rightSymbol": "%",
                            "values": [
                                {
                                    "x": "2023-09-30",
                                    "y": 50000000,
                                    "valueType": "CURRENCY"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    def multiple_companies_data(self, params: dict):
        """
        Fetch company data for multiple tickers and their associated metrics.

        Request body:
            {
                "tickers": [
                    "AAPL",
                    "MSFT"
                ]
            }

        Parameters:
            - metricName: The name of the metric to retrieve for each company. If not provided, all metrics will be returned.
                - string, optional. (e.g. research_and_development_msd)
            - startDate: The start date for the data points (inclusive).
                - string($date), optional. (e.g. 2023-01-01)
            - endDate: The end date for the data points (inclusive).
                - string($date), optional. (e.g. 2023-12-31)
            - freq: Frequency of the data points. Options are annual, quarterly, or ttm. Defaults to quarterly.
                - string, optional. (e.g. annual, quarterly, ttm)
            - yoy: A flag to indicate if Year-over-Year (YoY) calculations should be included. If true, the response will include YoY percentage changes for the dataset.
                - boolean, optional. (e.g. False)
            - percentRevenue: If true, the response will also include % of revenue of the datapoint.
                - boolean, optional. (e.g. False)

        Errors:
            - 400: Bad Request - Tickers should be an array.
                - Response:
                    {
                        "error": "Tickers should be an array."
                    }
        """
        return params
