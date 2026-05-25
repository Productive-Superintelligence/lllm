# Financial Modeling Prep Proxy
# https://site.financialmodelingprep.com/developer/docs/stable

import datetime as dt
import os

from ... import utils as U
from ..base import BaseProxy, ProxyRegistrator

COMPANY_REMOVE_KEYS = [
    "price",
    "marketCap",
    "beta",
    "lastDividend",
    "range",
    "change",
    "changePercentage",
    "volume",
    "averageVolume",
    "image",
    "defaultImage",
]


@ProxyRegistrator(
    path="fmp",
    name="Financial Modeling Prep API",
    description="Stock Market API and Financial Statements API. FMP is your source for the most reliable and accurate Stock Market API and Financial Data API available. Whether you are looking for real-time stock prices, financial statements, or historical data, we offer a comprehensive solution to meet all your financial data needs.",
)
class FMPProxy(BaseProxy):
    """
    Financial Modeling Prep

    Stock Market API and Financial Statements API
    FMP is your source for the most reliable and accurate Stock Market API and Financial Data API available. Whether you're looking for real-time stock prices, financial statements, or historical data, we offer a comprehensive solution to meet all your financial data needs.
    """

    def __init__(self, cutoff_date: str = None, use_cache: bool = True, **kwargs):
        super().__init__(cutoff_date=cutoff_date, use_cache=use_cache, **kwargs)
        self.api_key_name = "apikey"
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/stable"
        self.enums = {
            "period": ["quarter", "annual"],
            "timeframe": ["1min", "5min", "15min", "30min", "1hour", "4hour", "1day"],
        }

    def _call_api(
        self, url: str, params: dict, endpoint_info: dict, headers: dict
    ) -> dict:
        """
        Helper method to call the API using the requests library and remove specified keys.

        Args:
            url (str): The API endpoint URL.
            params (dict): Query parameters.

        Returns:
            dict: The filtered JSON response.
        """
        if "limit" in params:  # patch for filtering
            params["limit"] += params.get("limit", 50)

        if self.cutoff_date is not None:
            if "to" in params:  # move to cutoff date if to is after cutoff date
                to = dt.datetime.strptime(params["to"], "%Y-%m-%d")
                if to > self.cutoff_date:
                    from_ = dt.datetime.strptime(params["from"], "%Y-%m-%d")
                    time_diff = to - from_
                    params["to"] = self.cutoff_date.strftime("%Y-%m-%d")
                    params["from"] = (self.cutoff_date - time_diff).strftime("%Y-%m-%d")

            # set default from and to
            if "from" not in params:
                params["from"] = self.cutoff_date.strftime("%Y-%m-%d")
            if "to" not in params:
                params["to"] = self.cutoff_date.strftime("%Y-%m-%d")

        response = U.call_api(url, params, headers, self.use_cache)
        return response

    ########################################
    ### Search Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-symbol",
        name="Stock Symbol Search API",
        description="Easily find the ticker symbol of any stock with the FMP Stock Symbol Search API. Search by company name or symbol across multiple global markets.",
        params={
            "query*": (str, "AAPL"),
            "limit": (int, 50),
            "exchange": (str, "NASDAQ"),
        },
        response=[
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "currency": "USD",
                "exchangeFullName": "NASDAQ Global Select",
                "exchange": "NASDAQ",
            },
        ],
    )
    def search_symbol(self, params: dict) -> dict:
        """
        The FMP Stock Symbol Search API allows users to quickly and efficiently locate stock ticker symbols. Whether you're searching for U.S. stocks, international equities, or ETFs, this API provides fast, reliable results. Key features include:

         - Simple Search: Enter a company name or ticker symbol to retrieve essential details like the symbol, company name, exchange, and currency.
         - Global Market Access: Search across major stock exchanges, including NASDAQ, NYSE, and more.
         - Accurate and Up-to-Date: The API delivers real-time results, ensuring you're always working with the latest ticker information.
        The Stock Symbol Search API is perfect for traders, investors, or anyone needing quick access to stock symbols across different markets.
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-name",
        name="Company Name Search API",
        description="Search for ticker symbols, company names, and exchange details for equity securities and ETFs listed on various exchanges with the FMP Name Search API. This endpoint is useful for retrieving ticker symbols when you know the full or partial company or asset name but not the symbol identifier.",
        params={"query*": (str, "AA"), "limit": (int, 50), "exchange": (str, "NASDAQ")},
        response=[
            {
                "symbol": "AAGUSD",
                "name": "AAG USD",
                "currency": "USD",
                "exchangeFullName": "CCC",
                "exchange": "CRYPTO",
            },
        ],
    )
    def search_name(self, params: dict) -> dict:
        """
        About Company Name Search API
        The FMP Name Search API provides an easy way to find the ticker symbols and exchange information for companies and ETFs. This endpoint is useful for retrieving ticker symbols when you know the company or asset name but not the symbol identifier.

        Key Features of the Name Search API

         - Simple Company Name Lookup: Enter a company or asset name, and retrieve the corresponding ticker symbol, company name, and exchange details.
         - Equity Securities and ETFs: Supports searches for a variety of listed equity securities and exchange-traded funds (ETFs) across major exchanges.
         - Accurate and Up-to-Date Data: Receive real-time, accurate search results, ensuring you're always working with the latest available information.

        How Investors and Analysts Can Benefit

         - Quick Symbol Lookup: Easily locate ticker symbols when you know the company name but not the corresponding symbol.
         - Broad Market Coverage: Search across multiple exchanges for both domestic and international companies, helping you stay informed about different markets.
         - Streamlined Workflow: Enhance your research and investment decisions by quickly identifying the correct symbols for analysis or trade execution.
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-cik",
        name="CIK API",
        description="Easily retrieve the Central Index Key (CIK) for publicly traded companies with the FMP CIK API. Access unique identifiers needed for SEC filings and regulatory documents for a streamlined compliance and financial analysis process.",
        params={"cik*": (str, "320193"), "limit": (int, 50)},
        response=[
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "cik": "0000320193",
                "exchangeFullName": "NASDAQ Global Select",
                "exchange": "NASDAQ",
                "currency": "USD",
            },
        ],
    )
    def search_cik(self, params: dict) -> dict:
        """
        About CIK API
        The FMP CIK API is an essential tool for financial professionals, compliance officers, and analysts who need to quickly and accurately retrieve the Central Index Key (CIK) for a specific company. The CIK is a unique identifier used by the U.S. Securities and Exchange Commission (SEC) to track company filings, making it crucial for accessing corporate disclosures and financial data.

        Key Features of the CIK API

        - Quick CIK Lookup: Retrieve a company’s CIK by entering its symbol or name, allowing for efficient access to SEC filings and other regulatory information.
        - Essential for Compliance: Ensure accurate and timely access to SEC filings for regulatory compliance and corporate governance purposes.
        - Comprehensive Market Coverage: Search for CIKs across companies listed on major U.S. stock exchanges like NASDAQ and the NYSE.
        The CIK API is invaluable for anyone dealing with corporate filings and compliance, providing seamless access to essential company identifiers.

        Example: Streamlined SEC Filings: A compliance officer can use the CIK API to quickly find a company’s CIK number and use it to retrieve all relevant SEC filings. This enables efficient monitoring of regulatory disclosures and financial statements.
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-cusip",
        name="CUSIP API",
        description="Easily search and retrieve financial securities information by CUSIP number using the FMP CUSIP API. Find key details such as company name, stock symbol, and market capitalization associated with the CUSIP.",
        remove_keys=["marketCap"],
        params={
            "cusip*": (str, "037833100"),
        },
        response=[
            {"symbol": "AAPL", "companyName": "Apple Inc.", "cusip": "037833100"}
        ],
    )
    def search_cusip(self, params: dict) -> dict:
        """
        About CUSIP API
        The FMP CUSIP API allows users to quickly retrieve comprehensive financial information linked to a specific CUSIP number (Committee on Uniform Securities Identification Procedures). This nine-character alphanumeric code uniquely identifies financial securities, making it an essential tool for investors, traders, and analysts.

        Key features of the CUSIP API include:

        - Accurate Identification: Find stock symbols and company names associated with specific CUSIP numbers, ensuring precise identification of securities.
        - Comprehensive Data: Retrieve relevant financial details, including market capitalization, alongside CUSIP and stock symbol information.
        - Versatility: The API supports various types of securities, including stocks, bonds, and mutual funds, offering a broad range of search capabilities across multiple financial markets.
        This API is a valuable resource for financial professionals who need to identify and analyze securities efficiently by their CUSIP.

        Example: A trader can use the CUSIP API to instantly locate the CUSIP number and market capitalization for Apple Inc. by simply searching for the stock symbol “AAPL,” streamlining the research process before executing a trade.
        """
        return params

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-isin",
        name="ISIN API",
        description="Easily search and retrieve the International Securities Identification Number (ISIN) for financial securities using the FMP ISIN API. Find key details such as company name, stock symbol, and market capitalization associated with the ISIN.",
        remove_keys=["marketCap"],
        params={
            "isin*": (str, "US0378331005"),
        },
        response=[{"symbol": "AAPL", "name": "Apple Inc.", "isin": "US0378331005"}],
    )
    def search_isin(self, params: dict) -> dict:
        """
        About ISIN API
        The FMP ISIN API allows users to quickly retrieve comprehensive financial information linked to a specific ISIN (International Securities Identification Number). This twelve-character alphanumeric code uniquely identifies financial securities globally, making it an essential tool for investors, traders, and financial analysts.

        Key features of the ISIN API include:

        - Accurate Identification: Quickly find stock symbols and company names associated with a specific ISIN, ensuring precise identification of global securities.
        - Comprehensive Data: Retrieve relevant financial details such as the company name, stock symbol, ISIN, and market capitalization.
        - Global Coverage: The ISIN API supports a wide range of international securities, including stocks, bonds, and mutual funds, offering a broad range of search capabilities across global markets.
        This API is a valuable resource for financial professionals needing to identify and analyze securities efficiently by their ISIN for global investments or research.

        Example: An investor can use the ISIN API to locate the ISIN and market capitalization for Apple Inc. by searching for the stock symbol “AAPL,” streamlining global investment research.
        """
        return params

    ########################################
    ### Analyst Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Analyst",
        endpoint="grades-historical",
        name="Historical Stock Grades API",
        sub_category="Upgrades Downgrades",
        description="Access a comprehensive record of analyst grades with the FMP Historical Grades API. This tool allows you to track historical changes in analyst ratings for specific stock symbol",
        params={"symbol*": (str, "AAPL"), "limit": (int, 100)},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-02-01",
                "analystRatingsBuy": 8,
                "analystRatingsHold": 14,
                "analystRatingsSell": 2,
                "analystRatingsStrongSell": 2,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def grades_historical(self, params: dict) -> dict:
        """
        About Historical Stock Grades API
        The FMP Historical Grades API offers an in-depth look at how analysts have rated specific stocks in the past. This API is perfect for:

        - Trend Analysis: Investors can use historical ratings to spot long-term trends in market sentiment for a stock, helping to predict future price movements.
        - Investment Strategy Optimization: By tracking changes in analyst sentiment over time, investors can adjust their strategies based on whether analysts are becoming more bullish or bearish.
        - Benchmarking Performance: Compare a stock’s historical ratings to its actual performance, enabling a deeper understanding of how well the stock has lived up to expectations.
        - Market Sentiment Tracking: Use the API to analyze how buy, hold, and sell ratings have changed, providing insight into broader market confidence or caution around a stock.
        This API empowers investors with historical context, offering a valuable tool for long-term financial analysis and decision-making.

        Example Use Case
        A portfolio manager can utilize the Historical Grades API to observe changes in analyst sentiment for a particular stock, helping them adjust their strategy based on evolving market outlooks.
        """
        return params

    ########################################
    ### Calendar Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Calendar",
        sub_category="Dividends",
        endpoint="dividends",
        name="Dividends Company API",
        description="Stay informed about upcoming dividend payments with the FMP Dividends Company API. This API provides essential dividend data for individual stock symbols, including record dates, payment dates, declaration dates, and more.",
        params={"symbol*": (str, "AAPL"), "limit": (int, 100)},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-02-10",
                "recordDate": "2022-02-10",
                "paymentDate": "2022-02-13",
                "declarationDate": "2022-01-30",
                "adjDividend": 0.25,
                "dividend": 0.25,
                "yield": 0.42955326460481097,
                "frequency": "Quarterly",
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def dividends(self, params: dict) -> dict:
        """
        About Dividends Company API
        The FMP Dividends Company API offers a comprehensive view of the dividend information for specific stocks. Designed for dividend-focused investors, this API delivers:

        - Dividend Schedule Overview: Get access to upcoming dividend details, including record date, payment date, and declaration date, to ensure timely information on dividend payouts.
        - Dividend Amount: View the dividend and adjusted dividend amounts to stay informed of expected payments.
        - Yield Data: Track the dividend yield for stocks to better assess the return on investment for dividend-focused portfolios.
        - Payment Frequency: Understand how often dividends are paid (e.g., quarterly, annually) to align your investment strategy with the stock’s payout schedule.
        With detailed dividend information such as the amount, adjusted dividend, yield, and payment frequency, investors can effectively plan around dividend schedules. This API is perfect for dividend investors who need up-to-date information to make informed decisions about their income-generating investments.

        Example Use Case
        A dividend investor can use the Dividends Company API to monitor Apple’s upcoming dividend payment, ensuring they hold the stock through the record date to receive the payment.
        """
        return params

    ########################################
    ### Chart Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-price-eod/light",
        name="Basic Stock Chart API",
        sub_category="End of Day",
        description="Access simplified stock chart data using the FMP Basic Stock Chart API. This API provides essential charting information, including date, price, and trading volume, making it ideal for tracking stock performance with minimal data and creating basic price and volume charts.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2023-02-04",
                "price": 232.8,
                "volume": 44489128,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_eod_light(self, params: dict) -> dict:
        """
        About Basic Stock Chart API
        The FMP Basic Stock Chart API delivers streamlined access to stock charting data for users who need to track price movements without overwhelming complexity. This API offers:

        - Date & Price Information: Easily track daily price movements for a specific stock symbol.
        - Volume Data: Stay informed about trading activity with volume data included for each date.
        - Basic Charting Needs: Ideal for generating simple stock price and volume charts for historical performance analysis.
        This API is perfect for users and developers who want a quick, straightforward way to visualize stock data without the need for detailed technical indicators.

        Example Use Case
        A financial app can use the Basic Stock Chart API to display a minimal chart showing a stock’s daily closing price and volume, allowing users to quickly assess its performance over time.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-price-eod/full",
        name="Stock Price and Volume Data API",
        sub_category="End of Day",
        description="Access full price and volume data for any stock symbol using the FMP Comprehensive Stock Price and Volume Data API. Get detailed insights, including open, high, low, close prices, trading volume, price changes, percentage changes, and volume-weighted average price (VWAP).",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2023-02-04",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "change": 5.6,
                "changePercent": 2.46479,
                "vwap": 230.86,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_eod_full(self, params: dict) -> dict:
        """
        About Stock Price and Volume Data API
        The FMP Comprehensive Stock Price and Volume Data API provides in-depth data on stock performance over time, making it an essential tool for analysts, traders, and investors. With this API, users can:

        - Detailed Price Data: Access complete price information, including opening, closing, high, and low prices for each trading day.
        - Trading Volume Insights: Retrieve data on daily trading volume to analyze liquidity and market activity.
        - Price Changes and Percentages: Track absolute price changes and percentage shifts to evaluate price movements.
        - VWAP (Volume-Weighted Average Price): Get the VWAP to measure the average price based on volume, helping to identify price trends and market behavior.
        This API is perfect for users who require detailed and accurate stock price and volume data to make informed trading and investment decisions.

        Example Use Case
        A financial analyst can use the Comprehensive Stock Price and Volume Data API to monitor Apple's daily stock performance, analyzing price changes, VWAP, and trading volume to spot trends and predict future price movements.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-price-eod/non-split-adjusted",
        name="Unadjusted Stock Price API",
        sub_category="End of Day",
        description="Access stock price and volume data without adjustments for stock splits with the FMP Unadjusted Stock Price Chart API. Get accurate insights into stock performance, including open, high, low, and close prices, along with trading volume, without split-related changes.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2023-02-04",
                "adjOpen": 227.2,
                "adjHigh": 233.13,
                "adjLow": 226.65,
                "adjClose": 232.8,
                "volume": 44489128,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_eod_non_split_adjusted(self, params: dict) -> dict:
        """
        About Unadjusted Stock Price API
        The FMP Unadjusted Stock Price Chart API provides unadjusted historical price data, allowing traders, analysts, and investors to view stock performance without split-related adjustments. This is useful for users who want a clear view of how stock prices moved before and after stock splits. Key features include:

        - Unadjusted Price Data: Access historical stock prices—open, high, low, and close—without any adjustments for stock splits.
        - Volume Data: Retrieve daily trading volume for further analysis of market activity.
        - Pre-Split Analysis: See how stock prices performed in their original form, making it easier to analyze trends prior to a split event.
        - Clear Historical View: For investors and analysts looking to avoid the distortions caused by stock splits, this API delivers clear and unmodified data.This API is ideal for anyone who needs accurate, split-free stock data for more precise historical analysis.

        Example Use Case
        A market researcher analyzing Apple stock performance before and after a split can use the Unadjusted Stock Price Chart API to get a clear view of stock prices without any split-related adjustments.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-price-eod/dividend-adjusted",
        name="Dividend Adjusted Price Chart API",
        sub_category="End of Day",
        description="Analyze stock performance with dividend adjustments using the FMP Dividend-Adjusted Price Chart API. Access end-of-day price and volume data that accounts for dividend payouts, offering a more comprehensive view of stock trends over time.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2023-02-04",
                "adjOpen": 227.2,
                "adjHigh": 233.13,
                "adjLow": 226.65,
                "adjClose": 232.8,
                "volume": 44489128,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_eod_dividend_adjusted(self, params: dict) -> dict:
        """
        About Dividend Adjusted Price Chart API
        The FMP Dividend-Adjusted Price Chart API delivers EOD (end-of-day) price data that is adjusted for dividends, helping traders, analysts, and investors understand stock performance while factoring in dividend payments. This ensures a more accurate analysis of stock value changes, particularly for companies with regular dividend payouts. Features include:

        - Dividend-Adjusted Prices: Access historical stock prices—open, high, low, and close—that have been adjusted for dividend payouts, reflecting the true stock value.
        - Volume Data: Retrieve daily trading volume to assess market activity alongside price movements.
        - Accurate Performance Analysis: Use dividend-adjusted data to evaluate a stock’s performance over time with the impact of dividends factored in.
        - Enhanced Historical Insights: Ideal for long-term investors who want a clearer picture of stock growth and performance, while including the effect of dividends.
        This API is a valuable tool for understanding total returns, making it easier to gauge a stock’s historical performance by incorporating dividend impacts.

        Example Use Case
        An investor tracking the historical growth of Apple stock can use the Dividend-Adjusted Price Chart API to account for the effect of dividend payouts when analyzing stock price changes over time.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/1min",
        name="1 Min Interval Stock Chart API",
        sub_category="Intraday",
        description="Access precise intraday stock price and volume data with the FMP 1-Minute Interval Stock Chart API. Retrieve real-time or historical stock data in 1-minute intervals, including key information such as open, high, low, and close prices, and trading volume for each minute.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 15:59:00",
                "open": 233.01,
                "low": 232.72,
                "high": 233.13,
                "close": 232.79,
                "volume": 720121,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_1min(self, params: dict) -> dict:
        """
        About 1 Min Interval Stock Chart API
        The FMP 1-Minute Interval Stock Chart API is designed for traders, analysts, and investors who need detailed intraday stock data for technical analysis, high-frequency trading, or algorithmic strategies. With this API, you can:

        - Detailed Intraday Data: Get stock prices at 1-minute intervals, including open, high, low, and close prices, as well as trading volume for each minute.
        - Real-Time and Historical Data: Access real-time minute-by-minute data or retrieve historical data using specific date ranges, allowing for long-term analysis.
        - Customization with Date Parameters: Easily pull data for any desired time frame, including historical data going back over 30 years, by setting the "from" and "to" parameters.
        - Intraday Charting: Perfect for building detailed intraday charts that provide deeper insights into short-term stock movements.
        - Perfect for Day Traders: For day traders or algorithmic traders, this API offers the precision needed to identify short-term trends, fluctuations, and trading opportunities.

        Example Use Case
        A day trader can use the 1-Minute Interval Stock Chart API to track Apple’s stock price movements throughout the trading day, enabling them to make timely buy and sell decisions based on real-time price changes and volume spikes.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/5min",
        name="5 Min Interval Stock Chart API",
        sub_category="Intraday",
        description="Access stock price and volume data with the FMP 5-Minute Interval Stock Chart API. Retrieve detailed stock data in 5-minute intervals, including open, high, low, and close prices, along with trading volume for each 5-minute period. This API is perfect for short-term trading analysis and building intraday charts.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 15:55:00",
                "open": 232.87,
                "low": 232.72,
                "high": 233.13,
                "close": 232.79,
                "volume": 1555040,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_5min(self, params: dict) -> dict:
        """
        About 5 Min Interval Stock Chart API
        The FMP 5-Minute Interval Stock Chart API provides users with valuable stock data over 5-minute intervals, allowing for better insight into intraday market activity. It's designed for investors and traders who need quick, accurate data to track short-term price movements. Key features include:

        - Short-Term Price Analysis: Track stock price movements over short periods with 5-minute interval data, providing an ideal solution for intraday traders.
        - Precise Trading Data: Get open, high, low, and close prices, along with trading volume, for each 5-minute period to identify patterns and trends.
        - Intraday Charting: Build detailed intraday charts for any stock symbol, allowing for enhanced visualization of short-term price trends.
        - Historical Data Access: Use the API to retrieve historical 5-minute interval data, providing a broader scope for price analysis and trend identification.
        Efficient for Active Traders: This API is perfect for day traders and active investors who need fast, reliable data to make informed trading decisions.

        Example Use Case
        A day trader can use the 5-Minute Interval Stock Chart API to monitor Apple's stock throughout the trading day, identifying short-term trends and making timely trading decisions based on price fluctuations.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/15min",
        name="15 Min Interval Stock Chart API",
        sub_category="Intraday",
        description="Access stock price and volume data with the FMP 15-Minute Interval Stock Chart API. Retrieve detailed stock data in 15-minute intervals, including open, high, low, close prices, and trading volume. This API is ideal for creating intraday charts and analyzing medium-term price trends during the trading day.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 15:45:00",
                "open": 232.25,
                "low": 232.18,
                "high": 233.13,
                "close": 232.79,
                "volume": 2535629,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_15min(self, params: dict) -> dict:
        """
        About 15 Min Interval Stock Chart API
        The FMP 15-Minute Interval Stock Chart API is designed to provide a more balanced view of stock price movements throughout the trading day. By delivering key data at 15-minute intervals, this API offers medium-term insights for traders and investors who need to monitor stock trends in a concise but effective format. Key features include:

        - Medium-Term Price Analysis: Monitor price fluctuations over 15-minute intervals, ideal for traders who need to identify intraday trends without analyzing every minute.
        - Comprehensive Data Points: Access key metrics such as open, high, low, close prices, and trading volume to create detailed intraday charts.
        - Flexible Intraday Monitoring: This API is suitable for traders and investors who need to track stock performance throughout the trading day, making it easier to spot price movements and trends.
        - Historical Data Access: Retrieve historical 15-minute interval data to conduct in-depth analysis of past trading sessions and identify recurring patterns.
        - Efficient Data Retrieval: Ideal for those who want a balance between fast-moving data (such as 1-minute intervals) and longer-term intraday data for smarter decision-making.

        Example Use Case
        A swing trader can use the 15-Minute Interval Stock Chart API to monitor Apple stock throughout the trading day, analyzing medium-term price movements to make strategic trade entries and exits based on significant fluctuations.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/30min",
        name="30 Min Interval Stock Chart API",
        sub_category="Intraday",
        description="Access stock price and volume data with the FMP 30-Minute Interval Stock Chart API. Retrieve essential stock data in 30-minute intervals, including open, high, low, close prices, and trading volume. This API is perfect for creating intraday charts and tracking medium-term price movements for more strategic trading decisions.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 15:30:00",
                "open": 232.29,
                "low": 232.01,
                "high": 233.13,
                "close": 232.79,
                "volume": 3476320,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_30min(self, params: dict) -> dict:
        """
        About 30 Min Interval Stock Chart API
        The FMP 30-Minute Interval Stock Chart API is designed for traders and investors seeking medium-term price insights without monitoring every minute of the trading day. By delivering key stock metrics in 30-minute intervals, it offers a well-balanced view of stock performance over time. Key features include:

        - Efficient Medium-Term Analysis: Monitor stock price fluctuations at 30-minute intervals, providing a clear view of price movements without the noise of smaller time frames.
        - Detailed Price Metrics: Access important data points such as open, high, low, close prices, and trading volume to build comprehensive intraday charts.
        - Ideal for Intraday Strategies: This API supports trading strategies that rely on medium-term price movements and volume patterns, making it ideal for day traders and investors.
        - Historical Data Availability: Retrieve historical data for 30-minute intervals, helping you analyze trends and patterns from past trading sessions.
        - Optimized for Trend Tracking: With data available at 30-minute intervals, this API offers an efficient solution for those looking to identify key trends during the trading day.

        Example Use Case
        A day trader uses the 30-Minute Interval Stock Chart API to monitor the performance of Apple stock over the course of a trading day, identifying important price patterns and volume changes to make calculated buy and sell decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/1hour",
        name="1 Hour Interval Stock Chart API",
        sub_category="Intraday",
        description="Track stock price movements over hourly intervals with the FMP 1-Hour Interval Stock Chart API. Access essential stock price and volume data, including open, high, low, and close prices for each hour, to analyze broader intraday trends with precision.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 15:30:00",
                "open": 232.29,
                "low": 232.01,
                "high": 233.13,
                "close": 232.79,
                "volume": 15079381,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_1hour(self, params: dict) -> dict:
        """
        About 1 Hour Interval Stock Chart API
        The FMP 1-Hour Interval Stock Chart API is perfect for traders and investors who want to monitor hourly stock price movements. By delivering key price metrics every hour, this API provides a clear and comprehensive view of intraday stock trends. Key features include:

        - Hourly Price Data: Access open, high, low, and close prices updated every hour to stay on top of stock performance throughout the trading day.
        - Volume Tracking: Get insights into hourly trading volumes to understand market activity and liquidity at different times of the day.
        - Broader Timeframe Analysis: Ideal for traders who focus on medium-to-long intraday trends, the API helps visualize price movements over a broader timeframe.
        - Historical Data: Retrieve hourly historical data to analyze past price performance and identify trends over time.
        - Ideal for Trend and Pattern Recognition: Use this data to identify key patterns such as support, resistance, or trend reversals over hourly intervals.

        Example Use Case
        A swing trader uses the 1-Hour Interval Stock Chart API to track the hourly performance of Apple stock throughout the day, helping them make informed buy and sell decisions based on observed trends and trading volume changes.
        """
        return params

    @BaseProxy.endpoint(
        category="Chart",
        endpoint="historical-chart/4hour",
        name="4 Hour Interval Stock Chart API",
        sub_category="Intraday",
        description="Analyze stock price movements over extended intraday periods with the FMP 4-Hour Interval Stock Chart API. Access key stock price and volume data in 4-hour intervals, perfect for tracking longer intraday trends and understanding broader market movements.",
        params={
            "symbol*": (str, "AAPL"),
            "from": ("date", "2022-11-04"),
            "to": ("date", "2023-02-04"),
            # "nonadjusted": (bool,False)
        },
        response=[
            {
                "date": "2023-02-04 12:30:00",
                "open": 231.79,
                "low": 231.37,
                "high": 233.13,
                "close": 232.37,
                "volume": 23781913,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_4hour(self, params: dict) -> dict:
        """
        About 4 Hour Interval Stock Chart API
        The FMP 4-Hour Interval Stock Chart API provides traders and investors with essential data points over longer intraday time frames, allowing for comprehensive trend analysis. Ideal for users who want to track price movements in blocks larger than 1 hour but still within the trading day. Key features include:

        - 4-Hour Price Intervals: Access open, high, low, and close prices, updated every 4 hours to provide a clearer view of intraday market trends.
        - Volume Data: Understand market activity by tracking trading volumes during each 4-hour period.
        - Ideal for Medium-Term Intraday Analysis: Longer intervals allow for deeper analysis of stock movements, helping to identify patterns and trends within a trading day.
        - Historical Data: Retrieve past 4-hour price data to study trends and create broader price movement models.
        - Intraday Market Strategy Support: Use the data to develop trading strategies that benefit from wider price movements and shifts within a trading session.
        Example Use Case
        A position trader uses the 4-Hour Interval Stock Chart API to monitor the longer intraday performance of Apple stock, allowing them to detect more substantial trends and price shifts without getting lost in short-term fluctuations.
        """
        return params

    ########################################
    ### Company Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Company",
        endpoint="profile",
        name="Company Profile Data API",
        sub_category="Profile",
        description=(
            "Access detailed company profile data with the FMP Company Profile Data API. This API provides key financial and operational information for a specific stock symbol, including the company's market capitalization, stock price, industry, and much more."
        ),
        remove_keys=COMPANY_REMOVE_KEYS,
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "currency": "USD",
                "cik": "0000320193",
                "isin": "US0378331005",
                "cusip": "037833100",
                "exchangeFullName": "NASDAQ Global Select",
                "exchange": "NASDAQ",
                "industry": "Consumer Electronics",
                "website": "https://www.apple.com",
                "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, ...",
                "ceo": "Mr. Timothy D. Cook",
                "sector": "Technology",
                "country": "US",
                "fullTimeEmployees": "164000",
                "phone": "(408) 996-1010",
                "address": "One Apple Park Way",
                "city": "Cupertino",
                "state": "CA",
                "zip": "95014",
                "ipoDate": "1980-12-12",
                "isEtf": False,
                "isActivelyTrading": True,
                "isAdr": False,
                "isFund": False,
            }
        ],
    )
    def profile(self, params: dict) -> dict:
        """
        About Company Profile Data API
        The FMP Company Profile Data API offers comprehensive insights into a company's financial status and operational details. This API is ideal for analysts, traders, and investors who need an in-depth look at a company’s core financial metrics and business information. Key features include:

        - Stock Price and Market Cap: Get the latest stock price and market capitalization for the requested symbol.
        - Company Details: Access information like company name, description, CEO, and industry classification
        - Financial Metrics: Track important financial metrics like dividend yield, stock beta, and trading range to assess performance and volatility.
        - Global Identifiers: Retrieve global financial identifiers such as CIK, ISIN, and CUSIP to ensure accurate tracking across platforms.
        - Contact Information: Obtain contact details like the company’s address, phone number, and website for direct reference.
        - IPO Data: Learn about the company's IPO date, sector, and whether it’s actively trading.

        Example Use Case
        An investor researching potential tech investments can use the Company Profile Data API to review the current financial health of Apple Inc., assess its performance, and explore key metrics like its stock range and market cap to inform buying or selling decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="profile-cik",
        name="Company Profile by CIK API",
        sub_category="Profile",
        description=(
            "Retrieve detailed company profile data by CIK (Central Index Key) with the FMP Company Profile by CIK API. This API allows users to search for companies using their unique CIK identifier and access a full range of company data, including stock price, market capitalization, industry, and much more."
        ),
        params={"cik*": (str, "320193")},
        remove_keys=COMPANY_REMOVE_KEYS,
        response=[
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "currency": "USD",
                "cik": "0000320193",
                "isin": "US0378331005",
                "cusip": "037833100",
                "exchangeFullName": "NASDAQ Global Select",
                "exchange": "NASDAQ",
                "industry": "Consumer Electronics",
                "website": "https://www.apple.com",
                "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, ...",
                "ceo": "Mr. Timothy D. Cook",
                "sector": "Technology",
                "country": "US",
                "fullTimeEmployees": "164000",
                "phone": "(408) 996-1010",
                "address": "One Apple Park Way",
                "city": "Cupertino",
                "state": "CA",
                "zip": "95014",
                "ipoDate": "1980-12-12",
                "isEtf": False,
                "isActivelyTrading": True,
                "isAdr": False,
                "isFund": False,
            }
        ],
    )
    def profile_cik(self, params: dict) -> dict:
        """
        About Company Profile by CIK API
        The FMP Company Profile by CIK API provides comprehensive company information for users who want to look up firms using the CIK code. Ideal for compliance officers, analysts, and investors, this API allows access to vital company details based on their CIK number. Key features include:

        - Company Lookup by CIK: Easily find companies using their Central Index Key for fast and accurate identification.
        - Stock Price & Market Cap: Get the most up-to-date stock price and market capitalization data for the requested company.
        - Comprehensive Financial Data: Access essential financial metrics like beta, dividend yield, and trading range to evaluate a company's performance.
        - Global Identifiers: Retrieve key identifiers such as CIK, ISIN, and CUSIP to streamline cross-platform tracking of companies.
        - Company Information: Get in-depth details on the company's business operations, CEO, sector, and contact information.
        - IPO & Industry Data: View company industry, sector, and IPO details to better understand its market position.
        Example Use Case
        A compliance officer conducting a regulatory review can use the Company Profile by CIK API to quickly retrieve comprehensive data on Apple Inc. using its unique CIK number, ensuring accuracy in cross-referencing the company across different databases.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="historical-employee-count",
        name="Company Historical Employee Count API",
        sub_category="Employee Count",
        description=(
            "Access historical employee count data for a company based on specific reporting periods. The FMP Company Historical Employee Count API provides insights into how a company’s workforce has evolved over time, allowing users to analyze growth trends and operational changes.",
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 100), "page": (int, 0)},
        response=[
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "acceptanceTime": "2022-11-01 06:01:36",
                "periodOfReport": "2022-09-28",
                "companyName": "Apple Inc.",
                "formType": "10-K",
                "filingDate": "2022-11-01",
                "employeeCount": 164000,
                "source": "https://www.sec.gov/Archives/edgar/data/320193/...",
            }
        ],
        dt_cutoff=("filingDate", "%Y-%m-%d"),
    )
    def employee_count(self, params: dict) -> dict:
        """
        About Company Historical Employee Count API
        The FMP Company Historical Employee Count API is designed for users who need to track workforce trends for a company across various reporting periods. This data is especially useful for analyzing long-term growth, staffing changes, and the relationship between workforce size and financial performance. Key features include:

        - Historical Employee Count: Retrieve workforce size over different periods to analyze growth or decline trends.
        - Report Periods: Gain insights into specific timeframes of employee data, tied to annual or quarterly financial reports.
        - Filing Date and Form Type: Understand when the employee data was reported, along with the corresponding SEC form type (e.g., 10-K).
        - Direct SEC Links: Access the original SEC filings for in-depth research and transparency.
        This API is ideal for HR analysts, investors, and business strategists who want to track workforce changes and assess their impact on company operations.

        Example Use Case
        A financial analyst can use the Company Historical Employee Count API to compare the employee count of Apple Inc. over a five-year period to evaluate how workforce changes correlate with revenue growth and market expansion.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="historical-market-capitalization",
        name="Historical Market Cap API",
        sub_category="Market Cap",
        description=(
            "Access historical market capitalization data for a company using the FMP Historical Market Capitalization API. This API helps track the changes in market value over time, enabling long-term assessments of a company's growth or decline.",
        ),
        params={
            "symbol*": (str, "AAPL"),
            "limit": (int, 100),
            "from": ("date", "2022-01-01"),
            "to": ("date", "2022-03-01"),
        },
        response=[{"symbol": "AAPL", "date": "2022-02-29", "marketCap": 2784608472000}],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_market_capitalization(self, params: dict) -> dict:
        """
        About Historical Market Cap API
        The FMP Historical Market Capitalization API allows users to retrieve past market cap data for any company listed in the database. Key features include:

        - Track Long-Term Performance: Retrieve historical market cap data to analyze how a company's value has evolved over time.
        - Identify Trends: Use historical data to spot trends, whether it's consistent growth, decline, or periods of volatility.
        - Informed Investment Decisions: Investors can use this data to evaluate a company's long-term performance and make more informed investment choices.
        This API is ideal for analysts, portfolio managers, and investors looking to assess a company’s growth trajectory or historical performance in the market.

        Example Use Case
        An investor looking to evaluate Apple's historical performance can use the Historical Market Capitalization API to retrieve past market cap data. This helps them understand how Apple's valuation has changed over time, identifying periods of growth or decline and comparing it with overall market conditions.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="mergers-acquisitions-search",
        name="Search Mergers & Acquisitions API",
        sub_category="Mergers",
        description=(
            "Search for specific mergers and acquisitions data with the FMP Search Mergers and Acquisitions API. Retrieve detailed information on M&A activity, including acquiring and targeted companies, transaction dates, and links to official SEC filings.",
        ),
        params={"name*": (str, "Apple")},
        response=[
            {
                "symbol": "PEGY",
                "companyName": "Pineapple Energy Inc.",
                "cik": "0000022701",
                "targetedCompanyName": "Communications Systems, Inc.",
                "targetedCik": "0000022701",
                "targetedSymbol": "JCS",
                "transactionDate": "2021-11-12",
                "acceptedDate": "2021-11-12 09:54:22",
                "link": "https://www.sec.gov/Archives/edgar/data/22701/...",
            }
        ],
        dt_cutoff=("transactionDate", "%Y-%m-%d"),
    )
    def mergers_acquisitions_search(self, params: dict) -> dict:
        """
        About Search Mergers & Acquisitions API
        The FMP Search Mergers and Acquisitions API allows users to find mergers and acquisitions by company name, enabling a deeper understanding of corporate activity. This API is useful for those needing detailed data on past and ongoing deals, including:

        - Company-Specific M&A Data: Search for M&A transactions involving specific companies, either as the acquirer or target.
        - Transaction Dates: Access the exact dates of the transactions for precise tracking.
        - Filing Links: Obtain links to official SEC documents for detailed information on the terms and conditions of the deal.
        This API is perfect for financial analysts, researchers, and corporate strategists who need comprehensive M&A data to inform business or investment decisions.

        Example Use Case
        A corporate strategist can use the Search Mergers and Acquisitions API to identify past acquisition targets of a competitor. This information can help shape competitive strategies or identify industry trends that may affect future business opportunities.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="governance-executive-compensation",
        name="Executive Compensation API",
        sub_category="Executive Compensation",
        description=(
            "Retrieve comprehensive compensation data for company executives with the FMP Executive Compensation API. This API provides detailed information on salaries, stock awards, total compensation, and other relevant financial data, including filing details and links to official documents.",
        ),
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "cik": "0000320193",
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "filingDate": "2022-01-10",
                "acceptedDate": "2022-01-10 16:31:18",
                "nameAndPosition": "Kate Adams Senior Vice President, General Counsel and Secretary",
                "year": 2022,
                "salary": 1000000,
                "bonus": 0,
                "stockAward": 22323641,
                "optionAward": 0,
                "incentivePlanCompensation": 3571150,
                "allOtherCompensation": 46914,
                "total": 26941705,
                "link": "https://www.sec.gov/Archives/edgar/data/320193/...",
            }
        ],
        dt_cutoff=("filingDate", "%Y-%m-%d"),
    )
    def governance_executive_compensation(self, params: dict) -> dict:
        """
        About Executive Compensation API
        The FMP Executive Compensation API is designed to give investors, analysts, and researchers a complete overview of executive compensation for publicly traded companies. This API is beneficial for:

        - Executive Salary & Benefits: Retrieve data on annual salaries, stock awards, bonuses, and incentive plans.
        - Comprehensive Compensation Breakdown: Access detailed reports on total compensation, including base pay and additional awards or incentives.
        - Filing Information: Includes key filing dates and direct links to SEC filings for deeper analysis of compensation packages.
        This API provides valuable insights into how company executives are compensated, helping users understand leadership incentives and assess company governance.

        Example Use Case
        A compensation analyst can use the Executive Compensation API to compare CEO pay across different companies, analyzing how various forms of compensation—such as salary, stock awards, and performance incentives—impact executive behavior and company performance.
        """
        return params

    @BaseProxy.endpoint(
        category="Company",
        endpoint="executive-compensation-benchmark",
        name="Executive Compensation Benchmark API",
        sub_category="Executive Compensation",
        description=(
            "Gain access to average executive compensation data across various industries with the FMP Executive Compensation Benchmark API. This API provides essential insights for comparing executive pay by industry, helping you understand compensation trends and benchmarks.",
        ),
        params={"year*": (str, "2022")},
        response=[
            {
                "industryTitle": "ABRASIVE, ASBESTOS & MISC NONMETALLIC MINERAL PRODS",
                "year": 2022,
                "averageCompensation": 694313.1666666666,
            }
        ],
        dt_cutoff=("year", "%Y"),
    )
    def executive_compensation_benchmark(self, params: dict) -> dict:
        """
        About Executive Compensation Benchmark API
        The FMP Executive Compensation Benchmark API is designed to help businesses, analysts, and compensation consultants assess how executive pay compares across industries. It’s ideal for:

        - Industry Benchmarking: Evaluate average executive compensation within specific industries to determine market rates.
        - Compensation Trends: Understand how executive pay varies across different sectors, providing valuable insights for salary negotiations or organizational planning.
        - Competitive Analysis: Compare compensation data by industry to ensure your company remains competitive in attracting top talent.
        This API provides a valuable resource for HR professionals, compensation analysts, and business leaders seeking to align executive pay with industry standards.

        Example Use Case
        An HR professional can use the Executive Compensation Benchmark API to compare the average pay for executives in the technology sector against those in the consumer goods sector, helping to determine competitive salary packages for their company's leadership team.
        """
        return params

    ########################################
    ### Commitment Of Traders Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Commitment Of Traders",
        endpoint="commitment-of-traders-analysis",
        name="COT Analysis By Dates API",
        description=(
            "Gain in-depth insights into market sentiment with the FMP COT Report Analysis API. Analyze the Commitment of Traders (COT) reports for a specific date range to evaluate market dynamics, sentiment, and potential reversals across various sectors."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "symbol": "B6",
                "date": "2022-02-27 00:00:00",
                "name": "British Pound (B6)",
                "sector": "CURRENCIES",
                "exchange": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
                "currentLongMarketSituation": 66.85,
                "currentShortMarketSituation": 33.15,
                "marketSituation": "Bullish",
                "previousLongMarketSituation": 67.97,
                "previousShortMarketSituation": 32.03,
                "previousMarketSituation": "Bullish",
                "netPostion": 46358,
                "previousNetPosition": 46312,
                "changeInNetPosition": 0.1,
                "marketSentiment": "Increasing Bullish",
                "reversalTrend": False,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def cot_analysis(self, params: dict) -> dict:
        """
        About COT Analysis By Dates API
        The FMP COT Report Analysis API is designed for traders, analysts, and market strategists to interpret the long and short positions of traders over time, helping to track sentiment trends and potential market shifts. This API includes:

        - Market Sentiment Evaluation: Analyze the bullish or bearish sentiment based on long and short positions, helping you gauge the current market situation.
        - Net Position Changes: Track changes in net positions to understand whether sentiment is becoming more bullish or bearish.
        - Historical Sentiment Comparison: Compare current market sentiment with previous periods to detect trends or potential reversal points in the market.
        This API enables market participants to make informed decisions by providing detailed insights into how traders are positioned in various markets and how sentiment evolves over time.

        Example Use Case
        A commodity trader can use the COT Report Analysis API to assess the bullish sentiment in the energy market by tracking changes in the net position of Brent crude oil traders, allowing them to refine their trading strategy accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Commitment Of Traders",
        endpoint="commitment-of-traders-list",
        name="COT Symbol List API",
        description=(
            "Access a comprehensive list of available Commitment of Traders (COT) reports by commodity or futures contract using the FMP COT Report List API. This API provides an overview of different market segments, allowing users to retrieve and explore COT reports for a wide variety of commodities and financial instruments."
        ),
        params={},
        response=[{"symbol": "NG", "name": "Natural Gas (NG)"}],
    )
    def cot_symbol_list(self, params: dict = None) -> dict:
        """
        About COT Symbol List API
        The COT Report List API is ideal for traders, analysts, and researchers who want to access a complete list of available COT reports for specific markets. This API includes:

        - Comprehensive Market Coverage: Retrieve a list of all available COT reports across various commodities, from energy to agricultural products.
        - Easy Market Segmentation: Identify the markets and futures contracts available for analysis in the Commitment of Traders report.
        - Symbol Identification: Easily locate the symbol associated with each commodity or contract, enabling streamlined queries and in-depth analysis.
        This API is useful for quickly identifying which COT reports are available and for what market segments, enabling more focused and effective market research.

        Example Use Case
        A trader looking to assess market sentiment in the natural gas market can use the COT Report List API to identify the relevant futures contract and pull detailed sentiment data from the associated COT report.
        """
        return params

    ########################################
    ### Economics Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Economics",
        endpoint="treasury-rates",
        name="Treasury Rates API",
        description=(
            "Access real-time and historical Treasury rates for all maturities with the FMP Treasury Rates API. Track key benchmarks for interest rates across the economy."
        ),
        params={"from": (str, "2022-01-01"), "to": (str, "2022-03-01")},
        response=[
            {
                "date": "2022-02-29",
                "month1": 5.53,
                "month2": 5.5,
                "month3": 5.45,
                "month6": 5.3,
                "year1": 5.01,
                "year2": 4.64,
                "year3": 4.43,
                "year5": 4.26,
                "year7": 4.28,
                "year10": 4.25,
                "year20": 4.51,
                "year30": 4.38,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def treasury_rates(self, params: dict) -> dict:
        """
        About Treasury Rates API
        The Treasury Rates API provides real-time and historical data on Treasury rates for all maturities. These rates represent the interest rates that the US government pays on its debt obligations and serve as a critical benchmark for interest rates across the economy. Investors can use this API to:

        - Track Treasury Rates Over Time: Monitor the movement of Treasury rates and understand how they change over different periods.
        - Identify Interest Rate Trends: Analyze trends in interest rates to gain insights into the broader economic landscape.
        - Make Informed Investment Decisions: Use the data to inform investment strategies based on current and historical interest rate information.
        This API is an invaluable tool for investors, analysts, and economists who need accurate and timely information on Treasury rates.
        """
        return params

    @BaseProxy.endpoint(
        category="Economics",
        endpoint="economic-indicators",
        name="Economic Indicators API",
        description=(
            "Access real-time and historical economic data for key indicators like GDP, unemployment, and inflation with the FMP Economic Indicators API. Use this data to measure economic performance and identify growth trends."
        ),
        params={
            "name*": (str, "GDP"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[{"name": "GDP", "date": "2022-01-01", "value": 28624.069}],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def economic_indicators(self, params: dict) -> dict:
        """
        About Economics Indicators API
        The FMP Economic Indicators API provides comprehensive access to real-time and historical data for a wide range of economic indicators, including GDP, unemployment rates, and inflation. These indicators are essential tools for:

        - Economic Performance Tracking: Economic indicators such as GDP, unemployment, and inflation provide a snapshot of the overall health of the economy. By tracking these indicators over time, investors and analysts can gauge economic performance and make predictions about future economic conditions.
        - Trend Identification: Identifying trends in economic growth is crucial for making informed investment decisions. The Economic Indicators API allows users to analyze historical data and detect patterns that can indicate economic expansion or contraction.
        - Informed Investment Decisions: Economic data is a key factor in making informed investment decisions. By understanding the current state of the economy and its trajectory, investors can better align their portfolios with economic cycles.
        Example Investor Use Case
        An investor might use the Economic Indicators API to monitor GDP growth rates over the past decade. By analyzing this data, the investor can identify periods of strong economic growth and align their investment strategy accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Economics",
        endpoint="economic-calendar",
        name="Economic Data Releases Calendar API",
        description=(
            "Stay informed with the FMP Economic Data Releases Calendar API. Access a comprehensive calendar of upcoming economic data releases to prepare for market impacts and make informed investment decisions."
        ),
        params={"from": (str, "2022-01-01"), "to": (str, "2022-03-01")},
        response=[
            {
                "date": "2022-03-01 03:35:00",
                "country": "JP",
                "event": "3-Month Bill Auction",
                "currency": "JPY",
                "previous": -0.112,
                "estimate": None,
                "actual": -0.096,
                "change": 0.016,
                "impact": "Low",
                "changePercentage": 14.286,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def economic_calendar(self, params: dict) -> dict:
        """
        About Economic Data Releases Calendar API
        The FMP Economic Data Releases Calendar API provides a detailed schedule of upcoming economic data releases. This tool is essential for investors who want to:

        - Stay Updated on Economic Events: Access a calendar that lists the dates and details of key economic data releases.
        - Prepare for Market Reactions: Anticipate market movements by staying informed about upcoming economic indicators and reports.
        - Make Informed Investment Decisions: Use the latest economic data to guide your investment strategies and decisions.
        This API is ideal for traders, analysts, and investors who need to stay ahead of market trends by monitoring critical economic data releases.
        """
        return params

    ########################################
    ### ESG Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="ESG",
        endpoint="esg-disclosures",
        name="ESG Investment Search API",
        description=(
            "Align your investments with your values using the FMP ESG Investment Search API. Discover companies and funds based on Environmental, Social, and Governance (ESG) scores, performance, controversies, and business involvement criteria."
        ),
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "date": "2022-12-28",
                "acceptedDate": "2023-01-30",
                "symbol": "AAPL",
                "cik": "0000320193",
                "companyName": "Apple Inc.",
                "formType": "8-K",
                "environmentalScore": 52.52,
                "socialScore": 45.18,
                "governanceScore": 60.74,
                "ESGScore": 52.81,
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000007/0000320193-25-000007-index.htm",
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def esg_investment_search(self, params: dict) -> dict:
        """
        About ESG Investment Search API
        The FMP ESG Investment Search API is designed to help investors find companies and funds that align with their Environmental, Social, and Governance (ESG) values. This powerful tool allows you to:

        - Search by ESG Scores: Identify companies and funds with strong ESG ratings that meet your investment criteria.
        - Evaluate Performance: Filter investments based on their ESG performance to ensure they align with your values and financial goals.
        - Assess Controversies: Avoid investments in companies involved in significant ESG controversies by filtering based on controversy scores.
        - Apply Business Involvement Screens: Screen companies and funds based on specific business activities or sectors that align with your ESG principles.
        Examples Use Cases

        - An investor focused on sustainability might search for companies with an ESG scores of 80 or higher to ensure strong environmental and social practices.
        - An investor concerned about environmental impact could search for companies with low ESG controversy scores to avoid potential risks.
        """
        return params

    @BaseProxy.endpoint(
        category="ESG",
        endpoint="esg-ratings",
        name="ESG Ratings API",
        description=(
            "Access comprehensive ESG ratings for companies and funds with the FMP ESG Ratings API. Make informed investment decisions based on environmental, social, and governance (ESG) performance data."
        ),
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "companyName": "Apple Inc.",
                "industry": "CONSUMER ELECTRONICS",
                "fiscalYear": 2022,
                "ESGRiskRating": "B",
                "industryRank": "4 out of 5",
            }
        ],
        dt_cutoff=("fiscalYear", "%Y"),
    )
    def esg_ratings(self, params: dict) -> dict:
        """
        About ESG Ratings API
        The FMP ESG Ratings API provides detailed ESG ratings for companies and funds, helping investors and analysts assess the sustainability and ethical impact of their investments. This API is essential for:

        - Evaluating ESG Performance: Access ESG ratings that reflect a company’s or fund’s performance across environmental, social, and governance criteria, sourced from corporate sustainability reports, ESG research firms, and government agencies.
        - Informed Investment Decisions: Use ESG ratings to identify companies and funds that align with your ethical and sustainability goals, ensuring that your investments support positive social and environmental outcomes.
        - Filtering Based on ESG Scores: Customize your search to filter for companies with high ESG ratings or low ESG controversy scores, helping you focus on organizations that meet your specific ESG criteria.
        This API is a valuable tool for socially conscious investors, financial analysts, and asset managers who prioritize ESG factors in their investment strategies.

        Examples Use Cases

        - High ESG Performance: An investor interested in companies with strong ESG practices can filter for those with an ESG rating of 80 or higher, ensuring that their investments align with their values.
        - Low ESG Controversy: An analyst focused on minimizing environmental risks in their portfolio may filter for companies with low ESG controversy scores, indicating fewer issues related to environmental or social impacts.
        """
        return params

    # @BaseProxy.endpoint(
    #     category='ESG',
    #     endpoint='esg-benchmark',
    #     name='ESG Benchmark Comparison API',
    #     description=(
    #         "Evaluate the ESG performance of companies and funds with the FMP ESG Benchmark Comparison API. Compare ESG leaders and laggards within industries to make informed and responsible investment decisions."
    #     ),
    #     params={
    #         "year*": (str,"2022")
    #     },
    #     response=[
    #         {
    #             "fiscalYear": 2022,
    #             "sector": "APPAREL RETAIL",
    #             "environmentalScore": 61.36,
    #             "socialScore": 67.44,
    #             "governanceScore": 68.1,
    #             "ESGScore": 65.63
    #         }
    #     ],
    #     dt_cutoff=('fiscalYear', '%Y')
    # )
    # def esg_benchmark_comparison(self, params: dict) -> dict:
    #     '''
    #     About ESG Benchmark Comparison API
    #     The FMP ESG Benchmark Comparison API allows investors and analysts to compare the Environmental, Social, and Governance (ESG) performance of companies and funds against their peers. This powerful tool helps you:

    #     - Identify ESG Leaders: Find companies and funds that excel in ESG performance by comparing them to industry peers.
    #     - Spot ESG Laggards: Identify companies that fall behind in ESG performance, allowing you to make informed decisions about where to allocate your investments.
    #     - Monitor ESG Improvements: Track companies that are making significant strides in their ESG ratings, signaling positive change and potential investment opportunities.
    #     Example Use Cases

    #     - For Investors: Filter for companies in the top 10% of their industry in ESG ratings to focus on industry leaders in sustainable practices.
    #     - For Analysts: Search for companies that have shown a significant increase in their ESG rating over the past year to identify those making notable improvements in their ESG performance.
    #     '''
    #     return params

    ########################################
    ### Statements Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial Statements",
        endpoint="income-statement",
        name="Real-Time Income Statement API",
        description=(
            "Access real-time income statement data for public companies, private companies, and ETFs with the FMP Real-Time Income Statements API. Track profitability, compare competitors, and identify business trends with up-to-date financial data."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "date": "2022-09-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-11-01",
                "acceptedDate": "2022-11-01 06:01:36",
                "fiscalYear": "2022",
                "period": "FY",
                "revenue": 391035000000,
                "costOfRevenue": 210352000000,
                "grossProfit": 180683000000,
                "researchAndDevelopmentExpenses": 31370000000,
                "generalAndAdministrativeExpenses": 0,
                "sellingAndMarketingExpenses": 0,
                "sellingGeneralAndAdministrativeExpenses": 26097000000,
                "otherExpenses": 0,
                "operatingExpenses": 57467000000,
                "costAndExpenses": 267819000000,
                "netInterestIncome": 0,
                "interestIncome": 0,
                "interestExpense": 0,
                "depreciationAndAmortization": 11445000000,
                "ebitda": 134661000000,
                "ebit": 123216000000,
                "nonOperatingIncomeExcludingInterest": 0,
                "operatingIncome": 123216000000,
                "totalOtherIncomeExpensesNet": 269000000,
                "incomeBeforeTax": 123485000000,
                "incomeTaxExpense": 29749000000,
                "netIncomeFromContinuingOperations": 93736000000,
                "netIncomeFromDiscontinuedOperations": 0,
                "otherAdjustmentsToNetIncome": 0,
                "netIncome": 93736000000,
                "netIncomeDeductions": 0,
                "bottomLineNetIncome": 93736000000,
                "eps": 6.11,
                "epsDiluted": 6.08,
                "weightedAverageShsOut": 15343783000,
                "weightedAverageShsOutDil": 15408095000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def income_statement(self, params: dict) -> dict:
        """
        About Real-Time Income Statement API
        The FMP Real-Time Income Statements API provides comprehensive access to income statement data for a wide range of companies, including public companies, private companies, and ETFs. This API is essential for:

        - Profitability Tracking: Monitor a company's revenue, expenses, and net income over time. The income statement, also known as the profit and loss statement, provides a detailed view of a company's financial performance during a specific period.
        - Competitive Analysis: Use the API to compare a company's financial performance to its competitors. By analyzing income statements across companies, investors can identify which businesses are leading in profitability and efficiency.
        - Trend Identification: Detect trends in a company's business by examining changes in revenue, expenses, and net income over multiple periods. This data is crucial for understanding a company's financial health and growth prospects.
        Example
        Financial Ratio Calculation: An investor can use the Real-Time Income Statements API to calculate key financial ratios, such as the price-to-earnings ratio (P/E ratio) and gross margin. These ratios help investors assess a company's valuation and profitability, enabling more informed investment decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial Statements",
        endpoint="balance-sheet-statement",
        name="Balance Sheet Data API",
        description=(
            "Access detailed balance sheet statements for publicly traded companies with the Balance Sheet Data API. Analyze assets, liabilities, and shareholder equity to gain insights into a company's financial health."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "date": "2022-09-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-11-01",
                "acceptedDate": "2022-11-01 06:01:36",
                "fiscalYear": "2022",
                "period": "FY",
                "cashAndCashEquivalents": 29943000000,
                "shortTermInvestments": 35228000000,
                "cashAndShortTermInvestments": 65171000000,
                "netReceivables": 66243000000,
                "accountsReceivables": 33410000000,
                "otherReceivables": 32833000000,
                "inventory": 7286000000,
                "prepaids": 0,
                "otherCurrentAssets": 14287000000,
                "totalCurrentAssets": 152987000000,
                "propertyPlantEquipmentNet": 45680000000,
                "goodwill": 0,
                "intangibleAssets": 0,
                "goodwillAndIntangibleAssets": 0,
                "longTermInvestments": 91479000000,
                "taxAssets": 19499000000,
                "otherNonCurrentAssets": 55335000000,
                "totalNonCurrentAssets": 211993000000,
                "otherAssets": 0,
                "totalAssets": 364980000000,
                "totalPayables": 95561000000,
                "accountPayables": 68960000000,
                "otherPayables": 26601000000,
                "accruedExpenses": 0,
                "shortTermDebt": 20879000000,
                "capitalLeaseObligationsCurrent": 1632000000,
                "taxPayables": 26601000000,
                "deferredRevenue": 8249000000,
                "otherCurrentLiabilities": 50071000000,
                "totalCurrentLiabilities": 176392000000,
                "longTermDebt": 85750000000,
                "deferredRevenueNonCurrent": 10798000000,
                "deferredTaxLiabilitiesNonCurrent": 0,
                "otherNonCurrentLiabilities": 35090000000,
                "totalNonCurrentLiabilities": 131638000000,
                "otherLiabilities": 0,
                "capitalLeaseObligations": 12430000000,
                "totalLiabilities": 308030000000,
                "treasuryStock": 0,
                "preferredStock": 0,
                "commonStock": 83276000000,
                "retainedEarnings": -19154000000,
                "additionalPaidInCapital": 0,
                "accumulatedOtherComprehensiveIncomeLoss": -7172000000,
                "otherTotalStockholdersEquity": 0,
                "totalStockholdersEquity": 56950000000,
                "totalEquity": 56950000000,
                "minorityInterest": 0,
                "totalLiabilitiesAndTotalEquity": 364980000000,
                "totalInvestments": 126707000000,
                "totalDebt": 106629000000,
                "netDebt": 76686000000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def balance_sheet_statement(self, params: dict) -> dict:
        """
        About Balance Sheet Data API
        The Balance Sheet Data API allows investors, analysts, and financial professionals to retrieve detailed balance sheet information for companies. This API is essential for:

        - Comprehensive Financial Analysis: View key data on assets, liabilities, and shareholder equity, allowing for a detailed assessment of a company's financial structure and solvency.
        - Evaluating Company Health: Determine a company's liquidity and leverage through short-term and long-term assets, liabilities, and shareholder equity positions.
        - Supporting Investment Decisions: Use the balance sheet to compare companies within the same industry or sector, ensuring you make informed investment decisions based on a company's financial stability.
        This API provides real-time and historical balance sheet data, offering a snapshot of a company's financial health over different periods. Whether you're analyzing a company's financial performance or conducting due diligence, this data helps you evaluate critical financial metrics with ease.

        Example Use Case
        An investor analyzing a potential stock purchase uses the Balance Sheet Data API to evaluate the company's assets and liabilities. They review how much cash the company has on hand, its debt obligations, and total equity to ensure the company is financially stable.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial Statements",
        endpoint="cash-flow-statement",
        name="Cash Flow Statement API",
        description=(
            "Gain insights into a company's cash flow activities with the Cash Flow Statements API. Analyze cash generated and used from operations, investments, and financing activities to evaluate the financial health and sustainability of a business."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "date": "2022-09-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-11-01",
                "acceptedDate": "2022-11-01 06:01:36",
                "fiscalYear": "2022",
                "period": "FY",
                "netIncome": 93736000000,
                "depreciationAndAmortization": 11445000000,
                "deferredIncomeTax": 0,
                "stockBasedCompensation": 11688000000,
                "changeInWorkingCapital": 3651000000,
                "accountsReceivables": -5144000000,
                "inventory": -1046000000,
                "accountsPayables": 6020000000,
                "otherWorkingCapital": 3821000000,
                "otherNonCashItems": -2266000000,
                "netCashProvidedByOperatingActivities": 118254000000,
                "investmentsInPropertyPlantAndEquipment": -9447000000,
                "acquisitionsNet": 0,
                "purchasesOfInvestments": -48656000000,
                "salesMaturitiesOfInvestments": 62346000000,
                "otherInvestingActivities": -1308000000,
                "netCashProvidedByInvestingActivities": 2935000000,
                "netDebtIssuance": -5998000000,
                "longTermNetDebtIssuance": -9958000000,
                "shortTermNetDebtIssuance": 3960000000,
                "netStockIssuance": -94949000000,
                "netCommonStockIssuance": -94949000000,
                "commonStockIssuance": 0,
                "commonStockRepurchased": -94949000000,
                "netPreferredStockIssuance": 0,
                "netDividendsPaid": -15234000000,
                "commonDividendsPaid": -15234000000,
                "preferredDividendsPaid": 0,
                "otherFinancingActivities": -5802000000,
                "netCashProvidedByFinancingActivities": -121983000000,
                "effectOfForexChangesOnCash": 0,
                "netChangeInCash": -794000000,
                "cashAtEndOfPeriod": 29943000000,
                "cashAtBeginningOfPeriod": 30737000000,
                "operatingCashFlow": 118254000000,
                "capitalExpenditure": -9447000000,
                "freeCashFlow": 108807000000,
                "incomeTaxesPaid": 26102000000,
                "interestPaid": 0,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def cash_flow_statement(self, params: dict) -> dict:
        """
        About Cash Flow Statement API
        The Cash Flow Statements API provides a detailed view of a company's cash flow, giving investors and analysts essential data to understand how a company generates and spends its cash. This API is critical for:

        - Assessing Financial Health: Evaluate a company’s ability to generate cash from its core operations and its reliance on investments and financing.
        - Understanding Cash Management: Track cash inflows and outflows from operating, investing, and financing activities to understand how well a company manages its cash resources.
        - Free Cash Flow Analysis: Analyze free cash flow to determine how much cash a company has left over after paying for capital expenditures, providing a clearer picture of financial flexibility.
        This API delivers real-time and historical cash flow data, offering a comprehensive look at how a company manages its cash, which is essential for investment decisions, financial modeling, and credit analysis.

        Example Use Case
        A financial analyst uses the Cash Flow Statements API to evaluate a company's operating cash flow and free cash flow, helping to assess whether the company can sustain operations, invest in growth, and return value to shareholders.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial StatementsTTM",
        endpoint="income-statement-ttm",
        name="Income Statements TTM API",
        description=(
            "Access a comprehensive set of trailing twelve-month (TTM) income statement data with the Income Statements TTM API."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5)},
        response=[
            {
                "date": "2022-12-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-01-31",
                "acceptedDate": "2022-01-31 06:01:27",
                "fiscalYear": "2022",
                "period": "Q1",
                "revenue": 395760000000,
                "costOfRevenue": 211657000000,
                "grossProfit": 184103000000,
                "researchAndDevelopmentExpenses": 31942000000,
                "generalAndAdministrativeExpenses": 0,
                "sellingAndMarketingExpenses": 0,
                "sellingGeneralAndAdministrativeExpenses": 26486000000,
                "otherExpenses": 0,
                "operatingExpenses": 58428000000,
                "costAndExpenses": 270085000000,
                "netInterestIncome": 0,
                "interestIncome": 0,
                "interestExpense": 0,
                "depreciationAndAmortization": 11677000000,
                "ebitda": 137352000000,
                "ebit": 125675000000,
                "nonOperatingIncomeExcludingInterest": 0,
                "operatingIncome": 125675000000,
                "totalOtherIncomeExpensesNet": 71000000,
                "incomeBeforeTax": 125746000000,
                "incomeTaxExpense": 29596000000,
                "netIncomeFromContinuingOperations": 96150000000,
                "netIncomeFromDiscontinuedOperations": 0,
                "otherAdjustmentsToNetIncome": 0,
                "netIncome": 96150000000,
                "netIncomeDeductions": 0,
                "bottomLineNetIncome": 96150000000,
                "eps": 6.31,
                "epsDiluted": 6.3,
                "weightedAverageShsOut": 15081724000,
                "weightedAverageShsOutDil": 15150865000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def income_statement_ttm(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial StatementsTTM",
        endpoint="balance-sheet-statement-ttm",
        name="Balance Sheet Statements TTM API",
        description=(
            "Retrieve trailing twelve-month (TTM) balance sheet data with the Balance Sheet Statements TTM API."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5)},
        response=[
            {
                "date": "2022-12-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-01-31",
                "acceptedDate": "2022-01-31 06:01:27",
                "fiscalYear": "2022",
                "period": "Q1",
                "cashAndCashEquivalents": 30299000000,
                "shortTermInvestments": 23476000000,
                "cashAndShortTermInvestments": 53775000000,
                "netReceivables": 59306000000,
                "accountsReceivables": 29639000000,
                "otherReceivables": 29667000000,
                "inventory": 6911000000,
                "prepaids": 0,
                "otherCurrentAssets": 13248000000,
                "totalCurrentAssets": 133240000000,
                "propertyPlantEquipmentNet": 46069000000,
                "goodwill": 0,
                "intangibleAssets": 0,
                "goodwillAndIntangibleAssets": 0,
                "longTermInvestments": 87593000000,
                "taxAssets": 0,
                "otherNonCurrentAssets": 77183000000,
                "totalNonCurrentAssets": 210845000000,
                "otherAssets": 0,
                "totalAssets": 344085000000,
                "totalPayables": 61910000000,
                "accountPayables": 61910000000,
                "otherPayables": 0,
                "accruedExpenses": 0,
                "shortTermDebt": 12843000000,
                "capitalLeaseObligationsCurrent": 0,
                "taxPayables": 0,
                "deferredRevenue": 8461000000,
                "otherCurrentLiabilities": 61151000000,
                "totalCurrentLiabilities": 144365000000,
                "longTermDebt": 83956000000,
                "deferredRevenueNonCurrent": 0,
                "deferredTaxLiabilitiesNonCurrent": 0,
                "otherNonCurrentLiabilities": 49006000000,
                "totalNonCurrentLiabilities": 132962000000,
                "otherLiabilities": 0,
                "capitalLeaseObligations": 0,
                "totalLiabilities": 277327000000,
                "treasuryStock": 0,
                "preferredStock": 0,
                "commonStock": 84768000000,
                "retainedEarnings": -11221000000,
                "additionalPaidInCapital": 0,
                "accumulatedOtherComprehensiveIncomeLoss": -6789000000,
                "otherTotalStockholdersEquity": 0,
                "totalStockholdersEquity": 66758000000,
                "totalEquity": 66758000000,
                "minorityInterest": 0,
                "totalLiabilitiesAndTotalEquity": 344085000000,
                "totalInvestments": 111069000000,
                "totalDebt": 96799000000,
                "netDebt": 66500000000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def balance_sheet_statement_ttm(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Financial StatementsTTM",
        endpoint="cash-flow-statement-ttm",
        name="Cashflow Statements TTM API",
        description=(
            "Access trailing twelve-month (TTM) cash flow statement data with the Cashflow Statements TTM API."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5)},
        response=[
            {
                "date": "2022-12-28",
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "cik": "0000320193",
                "filingDate": "2022-01-31",
                "acceptedDate": "2022-01-31 06:01:27",
                "fiscalYear": "2022",
                "period": "Q1",
                "netIncome": 96150000000,
                "depreciationAndAmortization": 11677000000,
                "deferredIncomeTax": 0,
                "stockBasedCompensation": 11977000000,
                "changeInWorkingCapital": -8224000000,
                "accountsReceivables": -9505000000,
                "inventory": -694000000,
                "accountsPayables": 3891000000,
                "otherWorkingCapital": -1916000000,
                "otherNonCashItems": -3286000000,
                "netCashProvidedByOperatingActivities": 108294000000,
                "investmentsInPropertyPlantAndEquipment": -9995000000,
                "acquisitionsNet": 0,
                "purchasesOfInvestments": -45000000000,
                "salesMaturitiesOfInvestments": 67422000000,
                "otherInvestingActivities": -1627000000,
                "netCashProvidedByInvestingActivities": 10800000000,
                "netDebtIssuance": -10967000000,
                "longTermNetDebtIssuance": -10967000000,
                "shortTermNetDebtIssuance": 0,
                "netStockIssuance": -98416000000,
                "netCommonStockIssuance": -98416000000,
                "commonStockIssuance": 0,
                "commonStockRepurchased": -98416000000,
                "netPreferredStockIssuance": 0,
                "netDividendsPaid": -15265000000,
                "commonDividendsPaid": -15265000000,
                "preferredDividendsPaid": 0,
                "otherFinancingActivities": -6121000000,
                "netCashProvidedByFinancingActivities": -130769000000,
                "effectOfForexChangesOnCash": 0,
                "netChangeInCash": -11675000000,
                "cashAtEndOfPeriod": 30299000000,
                "cashAtBeginningOfPeriod": 41974000000,
                "operatingCashFlow": 108294000000,
                "capitalExpenditure": -9995000000,
                "freeCashFlow": 98299000000,
                "incomeTaxesPaid": 37498000000,
                "interestPaid": 0,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def cash_flow_statement_ttm(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Ratios",
        endpoint="key-metrics",
        name="Key Metrics API",
        description=(
            "Access essential financial metrics for a company with the FMP Financial Key Metrics API. Evaluate revenue, net income, P/E ratio, and more to assess performance and compare it to competitors."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "limit": (int, 10),
            "period": (str, "annual"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "marketCap": 3495160329570,
                "enterpriseValue": 3571846329570,
                "evToSales": 9.134339201273542,
                "evToOperatingCashFlow": 30.204866893043786,
                "evToFreeCashFlow": 32.82735788662494,
                "evToEBITDA": 26.524727497716487,
                "netDebtToEBITDA": 0.5694744580836323,
                "currentRatio": 0.8673125765340832,
                "incomeQuality": 1.2615643936161134,
                "grahamNumber": 22.587017267616833,
                "grahamNetNet": -12.352478525015636,
                "taxBurden": 0.7590881483581001,
                "interestBurden": 1.0021831580314244,
                "workingCapital": -23405000000,
                "investedCapital": 22275000000,
                "returnOnAssets": 0.25682503150857583,
                "operatingReturnOnAssets": 0.3434290787011036,
                "returnOnTangibleAssets": 0.25682503150857583,
                "returnOnEquity": 1.6459350307287095,
                "returnOnInvestedCapital": 0.4430708117427921,
                "returnOnCapitalEmployed": 0.6533607652660827,
                "earningsYield": 0.026818798327209237,
                "freeCashFlowYield": 0.03113076074921754,
                "capexToOperatingCashFlow": 0.07988736110406414,
                "capexToDepreciation": 0.8254259501965924,
                "capexToRevenue": 0.02415896275269477,
                "salesGeneralAndAdministrativeToRevenue": 0,
                "researchAndDevelopementToRevenue": 0.08022299794136074,
                "stockBasedCompensationToRevenue": 0.02988990755303234,
                "intangiblesToTotalAssets": 0,
                "averageReceivables": 63614000000,
                "averagePayables": 65785500000,
                "averageInventory": 6808500000,
                "daysOfSalesOutstanding": 61.83255974529134,
                "daysOfPayablesOutstanding": 119.65847721913745,
                "daysOfInventoryOutstanding": 12.642570548414087,
                "operatingCycle": 74.47513029370543,
                "cashConversionCycle": -45.18334692543202,
                "freeCashFlowToEquity": 32121000000,
                "freeCashFlowToFirm": 117192805288.09166,
                "tangibleAssetValue": 56950000000,
                "netCurrentAssetValue": -155043000000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def key_metrics(self, params: dict) -> dict:
        """
        About Key Metrics API
        The FMP Financial Key Metrics API provides crucial financial data that helps investors, analysts, and managers assess a company’s financial performance. This endpoint offers:

        - Revenue: Track the total income generated by the company from its operations.
        - Net Income: Understand the company’s profitability after all expenses and taxes have been deducted.
        - P/E Ratio (Price-to-Earnings Ratio): Evaluate the company’s valuation relative to its earnings, helping to determine if the stock is overvalued or undervalued.
        These financial key performance indicators (KPIs) are invaluable tools for business analysis, goal tracking, and competitive benchmarking. By using these metrics, you can:

        - Assess Financial Performance: Get a clear picture of a company’s financial health and operational efficiency.
        - Compare to Competitors: Benchmark a company’s performance against its competitors to identify strengths, weaknesses, and market positioning.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Ratios",
        endpoint="ratios",
        name="Financial Ratios API",
        description=(
            "Analyze a company's financial performance using the Financial Ratios API. This API provides detailed profitability, liquidity, and efficiency ratios, enabling users to assess a company's operational and financial health across various metrics."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "limit": (int, 10),
            "period": (str, "annual"),
        },
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "grossProfitMargin": 0.4620634981523393,
                "ebitMargin": 0.31510222870075566,
                "ebitdaMargin": 0.3443707085043538,
                "operatingProfitMargin": 0.31510222870075566,
                "pretaxProfitMargin": 0.3157901466620635,
                "continuousOperationsProfitMargin": 0.23971255769943867,
                "netProfitMargin": 0.23971255769943867,
                "bottomLineProfitMargin": 0.23971255769943867,
                "receivablesTurnover": 5.903038811648023,
                "payablesTurnover": 3.0503480278422272,
                "inventoryTurnover": 28.870710952511665,
                "fixedAssetTurnover": 8.560310858143607,
                "assetTurnover": 1.0713874732862074,
                "currentRatio": 0.8673125765340832,
                "quickRatio": 0.8260068483831466,
                "solvencyRatio": 0.3414634938155374,
                "cashRatio": 0.16975259648963673,
                "priceToEarningsRatio": 37.287278415656736,
                "priceToEarningsGrowthRatio": -45.93792700808932,
                "forwardPriceToEarningsGrowthRatio": -45.93792700808932,
                "priceToBookRatio": 61.37243774486391,
                "priceToSalesRatio": 8.93822887866815,
                "priceToFreeCashFlowRatio": 32.12256867269569,
                "priceToOperatingCashFlowRatio": 29.55638142954995,
                "debtToAssetsRatio": 0.29215025480848267,
                "debtToEquityRatio": 1.872326602282704,
                "debtToCapitalRatio": 0.6518501763673821,
                "longTermDebtToCapitalRatio": 0.6009110021023125,
                "financialLeverageRatio": 6.408779631255487,
                "workingCapitalTurnoverRatio": -31.099932397502684,
                "operatingCashFlowRatio": 0.6704045534944896,
                "operatingCashFlowSalesRatio": 0.3024128274962599,
                "freeCashFlowOperatingCashFlowRatio": 0.9201126388959359,
                "debtServiceCoverageRatio": 5.024761722304708,
                "interestCoverageRatio": 0,
                "shortTermOperatingCashFlowCoverageRatio": 5.663777000814215,
                "operatingCashFlowCoverageRatio": 1.109022873702276,
                "capitalExpenditureCoverageRatio": 12.517624642743728,
                "dividendPaidAndCapexCoverageRatio": 4.7912969490701345,
                "dividendPayoutRatio": 0.16252026969360758,
                "dividendYield": 0.0043585983369965175,
                "dividendYieldPercentage": 0.43585983369965176,
                "revenuePerShare": 25.484914639368924,
                "netIncomePerShare": 6.109054070954992,
                "interestDebtPerShare": 6.949329249507765,
                "cashPerShare": 4.247388013764271,
                "bookValuePerShare": 3.711600978715614,
                "tangibleBookValuePerShare": 3.711600978715614,
                "shareholdersEquityPerShare": 3.711600978715614,
                "operatingCashFlowPerShare": 7.706965094592383,
                "capexPerShare": 0.6156891035281195,
                "freeCashFlowPerShare": 7.091275991064264,
                "netIncomePerEBT": 0.7590881483581001,
                "ebtPerEbit": 1.0021831580314244,
                "priceToFairValue": 61.37243774486391,
                "debtToMarketCap": 0.03050761336980449,
                "effectiveTaxRate": 0.24091185164189982,
                "enterpriseValueMultiple": 26.524727497716487,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def financial_ratios(self, params: dict) -> dict:
        """
        About Financial Ratios API
        The Financial Ratios API delivers key ratios that help investors, analysts, and researchers evaluate a company's performance. These ratios include profitability indicators like gross profit margin and net profit margin, liquidity metrics such as current ratio and quick ratio, and efficiency measurements like asset turnover and inventory turnover. This API offers a comprehensive view of a company's financial health and operational efficiency.

        - Profitability Ratios: Gain insight into a company's ability to generate profit, with metrics like net profit margin and return on equity.
        - Liquidity Ratios: Understand how well a company can meet its short-term obligations using ratios like current ratio and quick ratio.
        - Efficiency Ratios: Assess how effectively a company utilizes its assets with metrics such as asset turnover and inventory turnover.
        - Debt Ratios: Evaluate a company's leverage and debt management through ratios like debt-to-equity and interest coverage ratios.
        This API is an essential tool for investors and analysts looking to analyze financial ratios and make informed decisions based on a company's financial performance.

        Example Use Case
        A portfolio manager can use the Financial Ratios API to compare liquidity ratios between companies in the same industry, helping them identify firms with stronger financial stability and more efficient operations.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Analysis",
        endpoint="owner-earnings",
        name="Owner Earnings API",
        description=(
            "Retrieve a company's owner earnings with the Owner Earnings API, which provides a more accurate representation of cash available to shareholders by adjusting net income. This metric is crucial for evaluating a company’s profitability from the perspective of investors."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "limit": (int, 10),
        },
        response=[
            {
                "symbol": "AAPL",
                "reportedCurrency": "USD",
                "fiscalYear": "2022",
                "period": "Q1",
                "date": "2022-12-28",
                "averagePPE": 0.13969,
                "maintenanceCapex": -2279964750,
                "ownersEarnings": 27655035250,
                "growthCapex": -660035250,
                "ownersEarningsPerShare": 1.83,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def owner_earnings(self, params: dict) -> dict:
        """
        About Owner Earnings API
        The Owner Earnings API offers a detailed breakdown of a company’s cash flow adjusted for key factors, such as capital expenditures and depreciation. It is designed for:

        - Investor Evaluation: Calculate cash truly available to shareholders, giving a clearer picture of profitability beyond net income.
        - Valuation Analysis: Use owner earnings to make informed decisions when valuing a company for long-term investments.
        - Capex Insight: Get insights into both maintenance and growth capital expenditures (Capex) to assess how much of the company’s income is being reinvested.
        - Owner Earnings Per Share: Track the value available to each share, helping determine if a stock is a good investment.
        This API provides a robust view of a company’s profitability and cash flow potential, especially for value investors looking for long-term returns.

        Example Use Case
        An investor uses the Owner Earnings API to evaluate Apple’s true cash earnings before purchasing additional shares, ensuring that the company’s income aligns with their long-term investment strategy.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Analysis",
        endpoint="enterprise-values",
        name="Enterprise Values API",
        description=(
            "Access a company's enterprise value using the Enterprise Values API. This metric offers a comprehensive view of a company's total market value by combining both its equity (market capitalization) and debt, providing a better understanding of its worth."
        ),
        params={"symbol*": (str, "AAPL"), "period": (str, "annual"), "limit": (int, 1)},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "stockPrice": 227.79,
                "numberOfShares": 15343783000,
                "marketCapitalization": 3495160329570,
                "minusCashAndCashEquivalents": 29943000000,
                "addTotalDebt": 106629000000,
                "enterpriseValue": 3571846329570,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def enterprise_values(self, params: dict) -> dict:
        """
        About Enterprise Values API
        The Enterprise Values API provides key financial data to help assess a company’s value by including:

        - Market Capitalization: The total value of all outstanding shares based on the current stock price.
        - Debt & Cash: Includes total debt and subtracts cash and cash equivalents to get a full picture of a company’s financial standing.
        - Comprehensive Valuation: Enterprise value includes both equity and debt, making it a preferred measure for evaluating potential buyouts, mergers, or acquisitions.
        This API is ideal for analysts, investors, and finance professionals who need a complete understanding of a company’s valuation, especially when considering its overall market position.

        Example Use Case
        A financial analyst uses the Enterprise Values API to assess Apple’s total market value, factoring in debt and subtracting cash reserves, to determine whether it’s a good acquisition target.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Growth",
        endpoint="income-statement-growth",
        name="Income Statement Growth API",
        description=(
            "Track key financial growth metrics with the Income Statement Growth API. Analyze how revenue, profits, and expenses have evolved over time, offering insights into a company’s financial health and operational efficiency."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "growthRevenue": 0.020219940775141214,
                "growthCostOfRevenue": -0.017675600199872046,
                "growthGrossProfit": 0.06819471705252206,
                "growthGrossProfitRatio": 0.04776303446712012,
                "growthResearchAndDevelopmentExpenses": 0.04863780712017383,
                "growthGeneralAndAdministrativeExpenses": 0,
                "growthSellingAndMarketingExpenses": 0,
                "growthOtherExpenses": -1,
                "growthOperatingExpenses": 0.04776924900176856,
                "growthCostAndExpenses": -0.004331112631234571,
                "growthInterestIncome": -1,
                "growthInterestExpense": -1,
                "growthDepreciationAndAmortization": -0.006424168764649709,
                "growthEBITDA": 0.07026704816404387,
                "growthOperatingIncome": 0.07799581805933456,
                "growthIncomeBeforeTax": 0.08571604417246959,
                "growthIncomeTaxExpense": 0.7770145152619318,
                "growthNetIncome": -0.033599670086086914,
                "growthEPS": -0.008116883116883088,
                "growthEPSDiluted": -0.008156606851549727,
                "growthWeightedAverageShsOut": -0.02543458616683152,
                "growthWeightedAverageShsOutDil": -0.02557791606880283,
                "growthEBIT": 0.0471407082579099,
                "growthNonOperatingIncomeExcludingInterest": 1,
                "growthNetInterestIncome": 1,
                "growthTotalOtherIncomeExpensesNet": 1.4761061946902654,
                "growthNetIncomeFromContinuingOperations": -0.033599670086086914,
                "growthOtherAdjustmentsToNetIncome": 0,
                "growthNetIncomeDeductions": 0,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def income_statement_growth(self, params: dict) -> dict:
        """
        About Income Statement Growth API
        The Income Statement Growth API provides critical growth data, allowing users to track year-over-year changes in key income statement items, such as:

        - Revenue Growth: Monitor changes in a company’s total revenue, helping gauge overall business performance.
        - Profit Growth: Assess fluctuations in gross profit, operating income, and net income, offering insights into profitability trends.
        - Expense Growth: Analyze growth in operating expenses, cost of revenue, and specific line items like research and development or interest expenses.
        This API is a valuable tool for investors, analysts, and financial professionals who want to track a company's financial trends over time.

        Example Use Case
        A financial analyst can use the Income Statement Growth API to evaluate Apple’s revenue and net income trends over the past few years, identifying whether the company is experiencing consistent growth or declines in profitability.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Growth",
        endpoint="balance-sheet-statement-growth",
        name="Balance Sheet Statement Growth API",
        description=(
            "Analyze the growth of key balance sheet items over time with the Balance Sheet Statement Growth API. Track changes in assets, liabilities, and equity to understand the financial evolution of a company."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "growthCashAndCashEquivalents": -0.0007341898882029034,
                "growthShortTermInvestments": 0.11516302627413738,
                "growthCashAndShortTermInvestments": 0.058744212492892536,
                "growthNetReceivables": 0.08621792243994425,
                "growthInventory": 0.15084504817564365,
                "growthOtherCurrentAssets": -0.02776454576386526,
                "growthTotalCurrentAssets": 0.06562138667929733,
                "growthPropertyPlantEquipmentNet": -0.15992349565984992,
                "growthGoodwill": 0,
                "growthIntangibleAssets": 0,
                "growthGoodwillAndIntangibleAssets": 0,
                "growthLongTermInvestments": -0.09015953214513049,
                "growthTaxAssets": 0.09225857046829487,
                "growthOtherNonCurrentAssets": 0.5266933370120016,
                "growthTotalNonCurrentAssets": 0.014238076328719674,
                "growthOtherAssets": 0,
                "growthTotalAssets": 0.035160515396374756,
                "growthAccountPayables": 0.1014039066617687,
                "growthShortTermDebt": 0.32087050041121024,
                "growthTaxPayables": 2.01632838190271,
                "growthDeferredRevenue": 0.023322168465450935,
                "growthOtherCurrentLiabilities": -0.1254584832500786,
                "growthTotalCurrentLiabilities": 0.21391802240757563,
                "growthLongTermDebt": -0.10003043628845205,
                "growthDeferredRevenueNonCurrent": 0,
                "growthDeferredTaxLiabilitiesNonCurrent": 0,
                "growthOtherNonCurrentLiabilities": -0.09048495373370312,
                "growthTotalNonCurrentLiabilities": -0.09295867814151548,
                "growthOtherLiabilities": 0,
                "growthTotalLiabilities": 0.060574238130816666,
                "growthPreferredStock": 0,
                "growthCommonStock": 0.12821763398905328,
                "growthRetainedEarnings": -88.50467289719626,
                "growthAccumulatedOtherComprehensiveIncomeLoss": 0.3737338456164862,
                "growthOthertotalStockholdersEquity": 0,
                "growthTotalStockholdersEquity": -0.0836095645737457,
                "growthMinorityInterest": 0,
                "growthTotalEquity": -0.0836095645737457,
                "growthTotalLiabilitiesAndStockholdersEquity": 0.035160515396374756,
                "growthTotalInvestments": -0.04107194211936368,
                "growthTotalDebt": -0.0401393489845888,
                "growthNetDebt": -0.05469472282829777,
                "growthAccountsReceivables": 0.13223532601328453,
                "growthOtherReceivables": 0.04307907360930203,
                "growthPrepaids": 0,
                "growthTotalPayables": 0.5262653527335452,
                "growthOtherPayables": 0,
                "growthAccruedExpenses": 0,
                "growthCapitalLeaseObligationsCurrent": 0.03619047619047619,
                "growthAdditionalPaidInCapital": 0,
                "growthTreasuryStock": 0,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def balance_sheet_growth(self, params: dict) -> dict:
        """
        About Balance Sheet Statement Growth API
        The Balance Sheet Statement Growth API provides year-over-year growth metrics for key balance sheet components. This API is ideal for:

        - Asset Growth Analysis: Track changes in assets, such as cash, inventory, and long-term investments, to assess how a company’s resources are expanding or contracting.
        - Liability Growth Monitoring: Understand how short-term and long-term liabilities are evolving, including payables and debt.
        - Equity Growth Tracking: Monitor shifts in shareholder equity, retained earnings, and total equity, offering insights into a company’s financial health.
        This API helps financial analysts and investors evaluate a company's stability and growth by examining the evolution of its balance sheet items.

        Example Use Case
        An investor can use the Balance Sheet Statement Growth API to analyze how Apple’s cash reserves and debt levels have changed over the past year, helping them assess the company’s liquidity and financial health.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Growth",
        endpoint="cash-flow-statement-growth",
        name="Cash Flow Statement Growth API",
        description=(
            "Measure the growth rate of a company’s cash flow with the FMP Cashflow Statement Growth API. Determine how quickly a company’s cash flow is increasing or decreasing over time."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "growthNetIncome": -0.033599670086086914,
                "growthDepreciationAndAmortization": -0.006424168764649709,
                "growthDeferredIncomeTax": 0,
                "growthStockBasedCompensation": 0.07892550540016616,
                "growthChangeInWorkingCapital": 1.555116314429071,
                "growthAccountsReceivables": -2.0473933649289098,
                "growthInventory": 0.3535228677379481,
                "growthAccountsPayables": 4.1868713605082055,
                "growthOtherWorkingCapital": 2.4402563136072373,
                "growthOtherNonCashItems": -0.017512348450830714,
                "growthNetCashProvidedByOperatingActivites": 0.06975566069312394,
                "growthInvestmentsInPropertyPlantAndEquipment": 0.13796879277306323,
                "growthAcquisitionsNet": 0,
                "growthPurchasesOfInvestments": -0.6486294175448107,
                "growthSalesMaturitiesOfInvestments": 0.3698202750801951,
                "growthOtherInvestingActivites": 0.02169035153328347,
                "growthNetCashUsedForInvestingActivites": -0.2078272604588394,
                "growthDebtRepayment": -0.012662502110417018,
                "growthCommonStockIssued": 0,
                "growthCommonStockRepurchased": -0.2243584784010316,
                "growthDividendsPaid": -0.013910149750415973,
                "growthOtherFinancingActivites": 0.03493013972055888,
                "growthNetCashUsedProvidedByFinancingActivities": -0.12439163778482412,
                "growthEffectOfForexChangesOnCash": 0,
                "growthNetChangeInCash": -1.1378472222222222,
                "growthCashAtEndOfPeriod": -0.02583205908188828,
                "growthCashAtBeginningOfPeriod": 0.23061216319013492,
                "growthOperatingCashFlow": 0.06975566069312394,
                "growthCapitalExpenditure": 0.13796879277306323,
                "growthFreeCashFlow": 0.092615279562982,
                "growthNetDebtIssuance": 0.3942026057973942,
                "growthLongTermNetDebtIssuance": -0.6812426135404356,
                "growthShortTermNetDebtIssuance": 1.995475113122172,
                "growthNetStockIssuance": -0.2243584784010316,
                "growthPreferredDividendsPaid": -0.013910149750415973,
                "growthIncomeTaxesPaid": 0.3973981476524439,
                "growthInterestPaid": -1,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def cash_flow_growth(self, params: dict) -> dict:
        """
        About Cashflow Statement Growth API
        The FMP Cashflow Statement Growth API provides key insights into the cash flow growth rate of a company, an essential metric for assessing a company's financial health. This API is crucial for:

        - Financial Performance Evaluation: Analyze the rate at which a company’s cash flow is growing. A positive growth rate indicates that the company is generating more cash than it is using, which can signal strong financial health and operational efficiency.
        - Investment Decision-Making: Use cash flow growth data to identify companies with strong cash flow generation capabilities. Companies with consistent positive cash flow growth are often more stable and may represent good investment opportunities.
        - Risk Assessment: A negative cash flow growth rate can be a red flag, indicating that a company is using more cash than it is generating. This information can be used to evaluate the risk associated with investing in or continuing to hold a company’s stock.

        Example
        Investor Analysis: An investor might use the Cashflow Growth API to assess a manufacturing company’s financial health by examining its cash flow growth over the past five years. If the company shows consistent positive growth, the investor may decide to increase their investment in the company.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Growth",
        endpoint="financial-growth",
        name="Financial Statement Growth API",
        description=(
            "Analyze the growth of key financial statement items across income, balance sheet, and cash flow statements with the Financial Statement Growth API. Track changes over time to understand trends in financial performance."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 1), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "date": "2022-09-28",
                "fiscalYear": "2022",
                "period": "FY",
                "reportedCurrency": "USD",
                "revenueGrowth": 0.020219940775141214,
                "grossProfitGrowth": 0.06819471705252206,
                "ebitgrowth": 0.07799581805933456,
                "operatingIncomeGrowth": 0.07799581805933456,
                "netIncomeGrowth": -0.033599670086086914,
                "epsgrowth": -0.008116883116883088,
                "epsdilutedGrowth": -0.008156606851549727,
                "weightedAverageSharesGrowth": -0.02543458616683152,
                "weightedAverageSharesDilutedGrowth": -0.02557791606880283,
                "dividendsPerShareGrowth": 0.040371570095532654,
                "operatingCashFlowGrowth": 0.06975566069312394,
                "receivablesGrowth": 0.08621792243994425,
                "inventoryGrowth": 0.15084504817564365,
                "assetGrowth": 0.035160515396374756,
                "bookValueperShareGrowth": -0.059693251557224776,
                "debtGrowth": -0.0401393489845888,
                "rdexpenseGrowth": 0.04863780712017383,
                "sgaexpensesGrowth": 0.04672709770575967,
                "freeCashFlowGrowth": 0.092615279562982,
                "tenYRevenueGrowthPerShare": 2.3937532854122625,
                "fiveYRevenueGrowthPerShare": 0.8093292228858464,
                "threeYRevenueGrowthPerShare": 0.163506592883552,
                "tenYOperatingCFGrowthPerShare": 2.1417809176982403,
                "fiveYOperatingCFGrowthPerShare": 1.051533221923415,
                "threeYOperatingCFGrowthPerShare": 0.23720294833900227,
                "tenYNetIncomeGrowthPerShare": 2.76381558093543,
                "fiveYNetIncomeGrowthPerShare": 1.0421744314966246,
                "threeYNetIncomeGrowthPerShare": 0.07761907162786884,
                "tenYShareholdersEquityGrowthPerShare": -0.19003774225234785,
                "fiveYShareholdersEquityGrowthPerShare": -0.24235004889283715,
                "threeYShareholdersEquityGrowthPerShare": -0.017459858915902907,
                "tenYDividendperShareGrowthPerShare": 1.1722201809466772,
                "fiveYDividendperShareGrowthPerShare": 0.29890046876764864,
                "threeYDividendperShareGrowthPerShare": 0.14617932692103452,
                "ebitdaGrowth": None,
                "growthCapitalExpenditure": None,
                "tenYBottomLineNetIncomeGrowthPerShare": None,
                "fiveYBottomLineNetIncomeGrowthPerShare": None,
                "threeYBottomLineNetIncomeGrowthPerShare": None,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def financial_growth(self, params: dict) -> dict:
        """
        About Financial Statement Growth API
        The Financial Statement Growth API provides an overview of year-over-year growth in key financial metrics from income statements, balance sheets, and cash flow statements. It’s designed for analysts and investors who want to:

        - Assess Revenue Trends: See how a company's revenue has grown or contracted over time, highlighting overall business health.
        - Evaluate Profitability Growth: Track growth in net income, operating income, and EBIT to gauge profitability.
        - Monitor Asset & Debt Changes: Understand the growth or reduction in assets and liabilities, providing insights into financial management.
        - Examine Cash Flow Changes: View growth in operating cash flow and free cash flow to analyze liquidity and capital efficiency.
        This API helps in identifying long-term trends across financial statements, providing a comprehensive picture of a company's financial growth.

        Example Use Case
        An investor can use the Financial Statement Growth API to analyze Apple’s revenue, net income, and free cash flow growth over the past few years, helping them assess the company’s performance trends.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Formats",
        endpoint="financial-reports-dates",
        name="Financial Reports Dates API",
        description="Retrieve the dates for financial reports using the FMP Financial Reports Dates API.",
        params={"symbol*": (str, "AAPL")},
        remove_keys=["linkXlsx", "linkJson"],
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "Q1",
            }
        ],
        dt_cutoff=("fiscalYear", "%Y"),
    )
    def financial_reports_dates(self, params: dict) -> dict:
        """ """
        return params

    # @BaseProxy.endpoint(
    #     category='Statements',
    #     sub_category='Formats',
    #     endpoint='financial-reports-json',
    #     name='Financial Reports Form JSON API',
    #     description="Access comprehensive annual reports with the FMP Annual Reports on Form 10-K API. Obtain detailed information about a company’s financial performance, business operations, and risk factors as reported to the SEC.",
    #     params={
    #         "symbol*": (str,"AAPL"),
    #         "year*": (int,2022),
    #         "period*": (str,"annual")
    #     },
    #     response=[
    #         {
    #             "symbol": "AAPL",
    #             "period": "FY",
    #             "year": "2022",
    #             "Cover Page": [
    #                 {
    #                     "Cover Page - USD ($) shares in Thousands, $ in Millions": [
    #                         "12 Months Ended"
    #                     ]
    #                 },
    #                 {
    #                     "items": [
    #                         "Sep. 24, 2022",
    #                         "Oct. 14, 2022",
    #                         "Mar. 25, 2022"
    #                     ]
    #                 },
    #                 {
    #                     "Entity Information [Line Items]": [
    #                         " ",
    #                         " ",
    #                         " "
    #                     ]
    #                 }
    #             ],
    #             "Auditor Information": [
    #                 {
    #                     "Auditor Information": [
    #                         "12 Months Ended"
    #                     ]
    #                 },
    #                 {
    #                     "items": [
    #                         "Sep. 24, 2022"
    #                     ]
    #                 },
    #                 {
    #                     "Auditor Information [Abstract]": [
    #                         " "
    #                     ]
    #                 }
    #             ],
    #             "CONSOLIDATED STATEMENTS OF OPER": [
    #                 {
    #                     "CONSOLIDATED STATEMENTS OF OPERATIONS - USD ($) shares in Thousands, $ in Millions": [
    #                         "12 Months Ended"
    #                     ]
    #                 },
    #                 {
    #                     "items": [
    #                         "Sep. 24, 2022",
    #                         "Sep. 25, 2022",
    #                         "Sep. 26, 2022"
    #                     ]
    #                 },
    #                 {
    #                     "Net sales": [
    #                         394328,
    #                         365817,
    #                         274515
    #                     ]
    #                 }
    #             ],
    #             "... More sections": []
    #         }
    #     ],
    #     dt_cutoff=('year', '%Y')
    # )
    # def financial_reports_json(self, params: dict) -> dict:
    #     '''
    #     About Financial Reports Form 10-K JSON API
    #     The FMP Annual Reports on Form 10-K API provides investors, analysts, and researchers with direct access to the annual reports that public companies in the United States are required to file with the Securities and Exchange Commission (SEC). This API is an invaluable resource for:

    #     - In-Depth Financial Analysis: Access detailed financial statements and data included in a company's Form 10-K to evaluate its financial health and performance over the past fiscal year.
    #     - Understanding Business Operations: Gain insights into a company’s operations, including its business strategy, key markets, and operational challenges, as disclosed in the Form 10-K.
    #     - Assessing Risk Factors: Review the risk factors section of the Form 10-K to understand the potential challenges and uncertainties that a company faces, helping to inform your investment decisions.
    #     The FMP Annual Reports on Form 10-K API makes it easy to retrieve and analyze these comprehensive reports, providing a complete picture of a company's financial and operational status.
    #     '''
    #     return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Segmentation",
        endpoint="revenue-product-segmentation",
        name="Revenue Product Segmentation API",
        description=(
            "Access detailed revenue breakdowns by product line with the Revenue Product Segmentation API. Understand which products drive a company's earnings and get insights into the performance of individual product segments."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "period": (str, "annual"),
            "structure": (str, "flat"),
        },
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "FY",
                "reportedCurrency": None,
                "date": "2022-09-28",
                "data": {
                    "Mac": 29984000000,
                    "Service": 96169000000,
                    "Wearables, Home and Accessories": 37005000000,
                    "iPad": 26694000000,
                    "iPhone": 201183000000,
                },
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def revenue_product_segmentation(self, params: dict) -> dict:
        """
        About Revenue Product Segmentation API
        The Revenue Product Segmentation API provides a comprehensive breakdown of a company’s revenue by product, making it easy to analyze performance across different product categories. This API is ideal for:

        - Product-Specific Revenue Analysis: Understand how much each product contributes to the company’s total earnings.
        - Strategic Insights: Gain insights into the growth or decline of specific product segments to inform investment decisions or corporate strategy.
        - Competitive Benchmarking: Compare product segment revenues across different companies in the same industry to gauge market position.
        This API offers a detailed view of product-level revenue, helping users identify growth drivers and track the financial health of specific product lines.

        Example Use Case
        An investor can use the Revenue Product Segmentation API to see how much of Apple’s earnings come from iPhone sales compared to other products, such as Macs or wearables.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="Segmentation",
        endpoint="revenue-geographic-segmentation",
        name="Revenue Geographic Segments API",
        description=(
            "Access detailed revenue breakdowns by geographic region with the Revenue Geographic Segments API. Analyze how different regions contribute to a company’s total revenue and identify key markets for growth."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "period": (str, "annual"),
            "structure": (str, "flat"),
        },
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "FY",
                "reportedCurrency": None,
                "date": "2022-09-28",
                "data": {
                    "Americas Segment": 167045000000,
                    "Europe Segment": 101328000000,
                    "Greater China Segment": 66952000000,
                    "Japan Segment": 25052000000,
                    "Rest of Asia Pacific": 30658000000,
                },
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def revenue_geographic_segmentation(self, params: dict) -> dict:
        """
        About Revenue Geographic Segments API
        The Revenue Geographic Segments API allows users to retrieve revenue data segmented by geographical regions, helping investors and analysts understand the performance of a company in different markets. This API is ideal for:

        - Regional Revenue Analysis: Break down revenue contributions by geographical area to see which regions are driving growth.
        - Market Performance Insights: Analyze how a company is performing in key regions like the Americas, Europe, and Greater China.
        - Global Strategy Planning: For businesses, understanding geographic revenue distribution can help in developing regional strategies and identifying new opportunities for expansion.
        This API offers a granular view of regional revenue, making it easier to track a company’s global financial performance.

        Example Use Case
        An investor can use the Revenue Geographic Segments API to track Apple’s performance across key regions like the Americas, Europe, and Greater China, helping to identify emerging markets or regions with declining sales.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        sub_category="As Reported",
        endpoint="income-statement-as-reported",
        name="As Reported Income Statements API",
        description=(
            "Retrieve income statements as they were reported by the company with the As Reported Income Statements API. Access raw financial data directly from official company filings, including revenue, expenses, and net income."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "FY",
                "reportedCurrency": None,
                "date": "2022-09-27",
                "data": {
                    "revenuefromcontractwithcustomerexcludingassessedtax": 391035000000,
                    "costofgoodsandservicessold": 210352000000,
                    "grossprofit": 180683000000,
                    "researchanddevelopmentexpense": 31370000000,
                    "sellinggeneralandadministrativeexpense": 26097000000,
                    "operatingexpenses": 57467000000,
                    "operatingincomeloss": 123216000000,
                    "nonoperatingincomeexpense": 269000000,
                    "incometaxexpensebenefit": 29749000000,
                    "netincomeloss": 93736000000,
                    "earningspersharebasic": 6.11,
                    "earningspersharediluted": 6.08,
                    "weightedaveragenumberofsharesoutstandingbasic": 15343783000,
                    "weightedaveragenumberofdilutedsharesoutstanding": 15408095000,
                    "othercomprehensiveincomelossforeigncurrencytransactionandtranslationadjustmentnetoftax": 395000000,
                    "othercomprehensiveincomelossderivativeinstrumentgainlossbeforereclassificationaftertax": -832000000,
                    "othercomprehensiveincomelossderivativeinstrumentgainlossreclassificationaftertax": 1337000000,
                    "othercomprehensiveincomelossderivativeinstrumentgainlossafterreclassificationandtax": -2169000000,
                    "othercomprehensiveincomeunrealizedholdinggainlossonsecuritiesarisingduringperiodnetoftax": 5850000000,
                    "othercomprehensiveincomelossreclassificationadjustmentfromaociforsaleofsecuritiesnetoftax": -204000000,
                    "othercomprehensiveincomelossavailableforsalesecuritiesadjustmentnetoftax": 6054000000,
                    "othercomprehensiveincomelossnetoftaxportionattributabletoparent": 4280000000,
                    "comprehensiveincomenetoftax": 98016000000,
                },
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def income_statement_as_reported(self, params: dict) -> dict:
        """
        About As Reported Income Statements API
        The As Reported Income Statements API provides a clear and direct view of a company's financial performance as reported in their official financial statements. This API is useful for:

        - Direct Financial Insights: Access income statement data as reported by the company, without adjustments.
        - Comprehensive Expense Tracking: See detailed breakdowns of revenue, cost of goods sold, and operating expenses.
        - In-Depth Analysis: Use the raw data to perform your own calculations and build models based on official figures.
        This API allows investors and analysts to rely on the most accurate, company-provided financial information for evaluating profitability and operational efficiency.

        Example Use Case
        A financial analyst can use the As Reported Income Statements API to access Apple’s quarterly income statements, allowing them to compare operating income and net profit for different fiscal periods without any adjustments.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        endpoint="balance-sheet-statement-as-reported",
        name="As Reported Balance Statements API",
        description=(
            "Access balance sheets as reported by the company with the As Reported Balance Statements API. View detailed financial data on assets, liabilities, and equity directly from official filings."
        ),
        sub_category="As Reported",
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "FY",
                "reportedCurrency": None,
                "date": "2022-09-27",
                "data": {
                    "cashandcashequivalentsatcarryingvalue": 29943000000,
                    "marketablesecuritiescurrent": 35228000000,
                    "accountsreceivablenetcurrent": 33410000000,
                    "nontradereceivablescurrent": 32833000000,
                    "inventorynet": 7286000000,
                    "otherassetscurrent": 14287000000,
                    "assetscurrent": 152987000000,
                    "marketablesecuritiesnoncurrent": 91479000000,
                    "propertyplantandequipmentnet": 45680000000,
                    "otherassetsnoncurrent": 74834000000,
                    "assetsnoncurrent": 211993000000,
                    "assets": 364980000000,
                    "accountspayablecurrent": 68960000000,
                    "otherliabilitiescurrent": 78304000000,
                    "contractwithcustomerliabilitycurrent": 8249000000,
                    "commercialpaper": 10000000000,
                    "longtermdebtcurrent": 10912000000,
                    "liabilitiescurrent": 176392000000,
                    "longtermdebtnoncurrent": 85750000000,
                    "otherliabilitiesnoncurrent": 45888000000,
                    "liabilitiesnoncurrent": 131638000000,
                    "liabilities": 308030000000,
                    "commonstocksharesoutstanding": 15116786000,
                    "commonstocksharesissued": 15116786000,
                    "commonstocksincludingadditionalpaidincapital": 83276000000,
                    "retainedearningsaccumulateddeficit": -19154000000,
                    "accumulatedothercomprehensiveincomelossnetoftax": -7172000000,
                    "stockholdersequity": 56950000000,
                    "liabilitiesandstockholdersequity": 364980000000,
                    "commonstockparorstatedvaluepershare": 0.00001,
                    "commonstocksharesauthorized": 50400000000,
                },
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def balance_sheet_as_reported(self, params: dict) -> dict:
        """
        About As Reported Balance Statements API
        The As Reported Balance Statements API offers unadjusted balance sheet data as reported by companies. It provides insight into a company's financial position, including:

        - Asset Overview: View cash, receivables, inventory, and long-term assets as reported.
        - Liability Breakdown: Access current and non-current liabilities, deferred revenues, and more.
        - Equity Insights: Examine stockholders’ equity, including retained earnings and stock details.
        This API is ideal for analysts and investors who want raw, as-reported balance sheet data to perform accurate financial assessments.

        Example Use Case
        An investment analyst can use the As Reported Balance Statements API to evaluate Apple's asset-liability structure for Q1 2010, helping to understand the company's financial position during that period without any adjustments.
        """
        return params

    @BaseProxy.endpoint(
        category="Statements",
        endpoint="cash-flow-statement-as-reported",
        name="As Reported Cashflow Statements API",
        description=(
            "View cash flow statements as reported by the company with the As Reported Cash Flow Statements API. Analyze a company's cash flows related to operations, investments, and financing directly from official reports."
        ),
        sub_category="As Reported",
        params={"symbol*": (str, "AAPL"), "limit": (int, 5), "period": (str, "annual")},
        response=[
            {
                "symbol": "AAPL",
                "fiscalYear": 2022,
                "period": "FY",
                "reportedCurrency": None,
                "date": "2022-09-27",
                "data": {
                    "cashcashequivalentsrestrictedcashandrestrictedcashequivalents": 29943000000,
                    "netincomeloss": 93736000000,
                    "depreciationdepletionandamortization": 11445000000,
                    "sharebasedcompensation": 11688000000,
                    "othernoncashincomeexpense": 2266000000,
                    "increasedecreaseinaccountsreceivable": 3788000000,
                    "increasedecreaseinotherreceivables": 1356000000,
                    "increasedecreaseininventories": 1046000000,
                    "increasedecreaseinotheroperatingassets": 11731000000,
                    "increasedecreaseinaccountspayable": 6020000000,
                    "increasedecreaseinotheroperatingliabilities": 15552000000,
                    "netcashprovidedbyusedinoperatingactivities": 118254000000,
                    "paymentstoacquireavailableforsalesecuritiesdebt": 48656000000,
                    "proceedsfrommaturitiesprepaymentsandcallsofavailableforsalesecurities": 51211000000,
                    "proceedsfromsaleofavailableforsalesecuritiesdebt": 11135000000,
                    "paymentstoacquirepropertyplantandequipment": 9447000000,
                    "paymentsforproceedsfromotherinvestingactivities": 1308000000,
                    "netcashprovidedbyusedininvestingactivities": 2935000000,
                    "paymentsrelatedtotaxwithholdingforsharebasedcompensation": 5600000000,
                    "paymentsofdividends": 15234000000,
                    "paymentsforrepurchaseofcommonstock": 94949000000,
                    "repaymentsoflongtermdebt": 9958000000,
                    "proceedsfromrepaymentsofcommercialpaper": 3960000000,
                    "proceedsfrompaymentsforotherfinancingactivities": -361000000,
                    "netcashprovidedbyusedinfinancingactivities": -121983000000,
                    "cashcashequivalentsrestrictedcashandrestrictedcashequivalentsperiodincreasedecreaseincludingexchangerateeffect": -794000000,
                    "incometaxespaidnet": 26102000000,
                },
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def cash_flow_as_reported(self, params: dict) -> dict:
        """
        About As Reported Cashflow Statements API
        The As Reported Cash Flow Statements API provides access to unadjusted cash flow data as reported by companies. This includes:

        - Operational Cash Flows: Examine the cash generated or used in day-to-day business activities.
        - Investment Cash Flows: Access cash movements related to investments in assets, acquisitions, and securities.
        - Financing Cash Flows: View cash from equity, debt issuance, and dividend payments.
        This API is ideal for users looking for a clear understanding of a company's cash flow management based on official filings.

        Example Use Case
        A financial analyst can use this API to track Apple's cash flow trends during Q1 2010, helping assess how effectively the company is managing its cash for operations and investments.
        """
        return params

    ########################################
    ### Form 13F Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Extract",
        endpoint="institutional-ownership/extract",
        name="SEC Filings Extract API",
        description=(
            "The SEC Filings Extract API allows users to extract detailed data directly from official SEC filings. This API provides access to key information such as company shares, security details, and filing links, making it easier to analyze corporate disclosures."
        ),
        params={
            "cik*": (str, "0001388838"),
            "year*": (str, "2022"),
            "quarter*": (str, "3"),
        },
        response=[
            {
                "date": "2022-09-30",
                "filingDate": "2022-11-13",
                "acceptedDate": "2022-11-13",
                "cik": "0001388838",
                "securityCusip": "674215207",
                "symbol": "CHRD",
                "nameOfIssuer": "CHORD ENERGY CORPORATION",
                "shares": 13280,
                "titleOfClass": "COM NEW",
                "sharesType": "SH",
                "putCallShare": "",
                "value": 2152290,
                "link": "https://www.sec.gov/Archives/edgar/data/1388838/000117266123003760/0001172661-23-003760-index.htm",
                "finalLink": "https://www.sec.gov/Archives/edgar/data/1388838/000117266123003760/infotable.xml",
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def sec_filings_extract(self, params: dict) -> dict:
        """
        About SEC Filings Extract API
        The SEC Filings Extract API offers a streamlined way to retrieve detailed information from SEC filings. This is ideal for investors, analysts, and financial professionals who need to analyze official company reports and gain insights into ownership structures, security details, and other critical data.
        This API is perfect for:

        - SEC Filings Analysis: Extract key information from SEC filings, such as shares owned, value, and security details.
        - Ownership Tracking: Monitor changes in company ownership over time by accessing filed reports.
        - Filing Comparison: Compare detailed data from different filing periods to track trends and changes.
        This API provides a structured and simplified way to access complex SEC filings data, helping you save time and focus on the analysis.

        Example Use Case
        An investment firm uses the SEC Filings Extract API to track changes in ownership for a specific company by extracting data from quarterly 13F filings. This helps the firm identify trends and adjust its investment strategy accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Extract",
        endpoint="institutional-ownership/dates",
        name="Form 13F Filings Dates API",
        description=(
            "The Form 13F Filings Dates API allows you to retrieve dates associated with Form 13F filings by institutional investors. This is crucial for tracking stock holdings of institutional investors at specific points in time, providing valuable insights into their investment strategies."
        ),
        params={"cik*": (str, "0001067983")},
        response=[{"date": "2022-09-30", "year": 2022, "quarter": 3}],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def form_13f_filings_dates(self, params: dict) -> dict:
        """
        About Form 13F Filings Dates API
        The Form 13F Filings Dates API is ideal for users interested in tracking when institutional investors file Form 13F reports with the SEC. This data reveals their stock holdings and investment trends, helping investors and analysts understand what major institutions are investing in during specific quarters.
        This API is perfect for:

        - Investor Monitoring: Track when institutional investors file their stock holdings with the SEC.
        - Quarterly Analysis: Review changes in institutional holdings across different quarters.
        - Historical Research: Analyze filing patterns over the years and spot trends in institutional ownership.
        This API provides a streamlined way to track the timing of institutional holdings, which is useful for investment analysis and understanding market trends.

        Example Use Case
        An analyst can use the Form 13F Filings Dates API to check the filing dates of a major institutional investor, allowing them to compare portfolio changes from quarter to quarter and make informed decisions based on institutional behavior.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Holder",
        endpoint="institutional-ownership/extract-analytics/holder",
        name="Filings Extract With Analytics By Holder API",
        description=(
            "The Filings Extract With Analytics By Holder API provides an analytical breakdown of institutional filings. This API offers insight into stock movements, strategies, and portfolio changes by major institutional holders, helping you understand their investment behavior and track significant changes in stock ownership."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "year*": (str, "2022"),
            "quarter*": (str, "3"),
        },
        response=[
            {
                "date": "2022-09-30",
                "cik": "0000102909",
                "filingDate": "2022-12-18",
                "investorName": "VANGUARD GROUP INC",
                "symbol": "AAPL",
                "securityName": "APPLE INC",
                "typeOfSecurity": "COM",
                "securityCusip": "037833100",
                "sharesType": "SH",
                "putCallShare": "Share",
                "investmentDiscretion": "SOLE",
                "industryTitle": "ELECTRONIC COMPUTERS",
                "weight": 5.4673,
                "lastWeight": 5.996,
                "changeInWeight": -0.5287,
                "changeInWeightPercentage": -8.8175,
                "marketValue": 222572509140,
                "lastMarketValue": 252876459509,
                "changeInMarketValue": -30303950369,
                "changeInMarketValuePercentage": -11.9837,
                "sharesNumber": 1299997133,
                "lastSharesNumber": 1303688506,
                "changeInSharesNumber": -3691373,
                "changeInSharesNumberPercentage": -0.2831,
                "quarterEndPrice": 171.21,
                "avgPricePaid": 95.86,
                "isNew": False,
                "isSoldOut": False,
                "ownership": 8.3336,
                "lastOwnership": 8.305,
                "changeInOwnership": 0.0286,
                "changeInOwnershipPercentage": 0.3445,
                "holdingPeriod": 42,
                "firstAdded": "2013-06-30",
                "performance": -29671950396,
                "performancePercentage": -11.7338,
                "lastPerformance": 38078179274,
                "changeInPerformance": -67750129670,
                "isCountedForPerformance": True,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def filings_extract_analytics_holder(self, params: dict) -> dict:
        """
        About Filings Extract With Analytics By Holder API
        The Filings Extract With Analytics By Holder API allows users to extract detailed analytics from filings by institutional investors. It offers information such as shares held, changes in stock weight and market value, ownership percentages, and other important metrics that provide an analytical view of institutional investment strategies.

        - Institutional Investor Analysis: Track the behavior of large institutional holders such as Vanguard, including their changes in stock positions and market value.
        - Portfolio Movement Monitoring: Analyze stock movements and holding period data to see how long institutions have held a stock and when they increased or reduced their positions.
        - Investment Strategy Insights: Understand investment strategies by looking at changes in weight, market value, and ownership over time.
        This API offers granular insights into how institutions manage their portfolios, providing data to investors and analysts for deeper investment analysis.

        Example Use Case
        An investment analyst can use the Filings Extract With Analytics By Holder API to monitor Vanguard Group's activity in Apple Inc. stocks, seeing how much stock Vanguard holds, any changes in weight or market value, and when the stock was first added to their portfolio.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Holder",
        endpoint="institutional-ownership/holder-performance-summary",
        name="Holder Performance Summary API",
        description=(
            "The Holder Performance Summary API provides insights into the performance of institutional investors based on their stock holdings. This data helps track how well institutional holders are performing, their portfolio changes, and how their performance compares to benchmarks like the S&P 500."
        ),
        params={"cik*": (str, "0001067983"), "page": (int, 0)},
        response=[
            {
                "date": "2022-09-30",
                "cik": "0001067983",
                "investorName": "BERKSHIRE HATHAWAY INC",
                "portfolioSize": 40,
                "securitiesAdded": 3,
                "securitiesRemoved": 4,
                "marketValue": 266378900503,
                "previousMarketValue": 279969062343,
                "changeInMarketValue": -13590161840,
                "changeInMarketValuePercentage": -4.8542,
                "averageHoldingPeriod": 18,
                "averageHoldingPeriodTop10": 31,
                "averageHoldingPeriodTop20": 27,
                "turnover": 0.175,
                "turnoverAlternateSell": 13.9726,
                "turnoverAlternateBuy": 1.1974,
                "performance": 17707926874,
                "performancePercentage": 6.325,
                "lastPerformance": 38318168662,
                "changeInPerformance": -20610241788,
                "performance1year": 89877376224,
                "performancePercentage1year": 28.5368,
                "performance3year": 91730847239,
                "performancePercentage3year": 31.2597,
                "performance5year": 157058602844,
                "performancePercentage5year": 73.1617,
                "performanceSinceInception": 182067479115,
                "performanceSinceInceptionPercentage": 198.2138,
                "performanceRelativeToSP500Percentage": 6.325,
                "performance1yearRelativeToSP500Percentage": 28.5368,
                "performance3yearRelativeToSP500Percentage": 36.5632,
                "performance5yearRelativeToSP500Percentage": 36.1296,
                "performanceSinceInceptionRelativeToSP500Percentage": 37.0968,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def holder_performance_summary(self, params: dict) -> dict:
        """
        About Holder Performance Summary API
        The Holder Performance Summary API allows users to view performance metrics for institutional holders, such as market value changes, portfolio turnover, and relative performance against benchmarks. This API is ideal for:

        - Institutional Investor Analysis: Track how well institutional investors are performing based on stock picks, changes in holdings, and market value.
        - Portfolio Turnover Analysis: See how frequently an institution buys or sells securities, providing insights into their trading strategy.
        - Performance Benchmarking: Compare an institution's performance against the S&P 500 and other benchmarks over different timeframes (1 year, 3 years, 5 years).
        This API offers a comprehensive view of an institutional holder’s performance over time, helping investors and analysts track key players in the market.

        Example Use Case
        An investment manager can use the Holder Performance Summary API to analyze Berkshire Hathaway's performance over the last five years and compare it to the S&P 500, assessing how well their investment strategy has fared.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Holder",
        endpoint="institutional-ownership/holder-industry-breakdown",
        name="Holders Industry Breakdown API",
        description=(
            "The Holders Industry Breakdown API provides an overview of the sectors and industries that institutional holders are investing in. This API helps analyze how institutional investors distribute their holdings across different industries and track changes in their investment strategies over time."
        ),
        params={
            "cik*": (str, "0001067983"),
            "year*": (str, "2022"),
            "quarter*": (str, "3"),
        },
        response=[
            {
                "date": "2022-09-30",
                "cik": "0001067983",
                "investorName": "BERKSHIRE HATHAWAY INC",
                "industryTitle": "ELECTRONIC COMPUTERS",
                "weight": 49.7704,
                "lastWeight": 51.0035,
                "changeInWeight": -1.2332,
                "changeInWeightPercentage": -2.4178,
                "performance": -20838154294,
                "performancePercentage": -178.2938,
                "lastPerformance": 26615340304,
                "changeInPerformance": -47453494598,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def holders_industry_breakdown(self, params: dict) -> dict:
        """
        About Holders Industry Breakdown API
        The Holders Industry Breakdown API allows users to retrieve data on the industries institutional investors are focusing on, including the weight of their holdings in each sector and how that weight changes over time. This API provides detailed insights into the industry allocation of institutional investors, making it easier to understand their sector focus and strategy.

        - Industry Focus Analysis: Understand which industries are receiving the most investment from major institutional holders.
        - Portfolio Diversification: Analyze how diversified institutional investors' portfolios are across different sectors.
        - Investment Trend Insights: Track changes in the weight of industry holdings to identify shifts in institutional investment strategies.
        This API is ideal for investors, analysts, and portfolio managers looking to gain insights into institutional investment behavior across various industries.

        Example Use Case
        A financial analyst can use the Holders Industry Breakdown API to analyze Berkshire Hathaway's sector focus, identifying whether they are increasing or decreasing their exposure to industries like technology or healthcare over time.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Symbol",
        endpoint="institutional-ownership/symbol-positions-summary",
        name="Positions Summary API",
        description=(
            "The Positions Summary API provides a comprehensive snapshot of institutional holdings for a specific stock symbol. It tracks key metrics like the number of investors holding the stock, changes in the number of shares, total investment value, and ownership percentages over time."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "year*": (str, "2022"),
            "quarter*": (str, "3"),
        },
        response=[
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "date": "2022-09-30",
                "investorsHolding": 4805,
                "lastInvestorsHolding": 4749,
                "investorsHoldingChange": 56,
                "numberOf13Fshares": 9247670386,
                "lastNumberOf13Fshares": 9345671472,
                "numberOf13FsharesChange": -98001086,
                "totalInvested": 1613733330618,
                "lastTotalInvested": 1825154796061,
                "totalInvestedChange": -211421465443,
                "ownershipPercent": 59.2821,
                "lastOwnershipPercent": 59.5356,
                "ownershipPercentChange": -0.2535,
                "newPositions": 158,
                "lastNewPositions": 188,
                "newPositionsChange": -30,
                "increasedPositions": 1921,
                "lastIncreasedPositions": 1775,
                "increasedPositionsChange": 146,
                "closedPositions": 156,
                "lastClosedPositions": 122,
                "closedPositionsChange": 34,
                "reducedPositions": 2375,
                "lastReducedPositions": 2506,
                "reducedPositionsChange": -131,
                "totalCalls": 173528138,
                "lastTotalCalls": 198746782,
                "totalCallsChange": -25218644,
                "totalPuts": 192878290,
                "lastTotalPuts": 177007062,
                "totalPutsChange": 15871228,
                "putCallRatio": 1.1115,
                "lastPutCallRatio": 0.8906,
                "putCallRatioChange": 22.0894,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def positions_summary(self, params: dict) -> dict:
        """
        About Positions Summary API
        The Positions Summary API enables users to analyze institutional positions in a particular stock by providing data such as the number of investors holding the stock, the number of shares held, the total amount invested, and changes in these metrics over a given time period. It is ideal for:

        - Tracking Institutional Investment Trends: Monitor how institutional investors are changing their positions in a stock over time.
        - Ownership Insights: Understand what percentage of a company is owned by institutional investors and how this changes.
        - Call & Put Analysis: Get insights into the put/call ratio and track options activity for institutional positions.
        This API is ideal for understanding institutional activity in the market and gaining insights into the behavior of major investors. It is essential for investors, analysts, and portfolio managers who want to keep a close eye on institutional movements in specific stocks.

        Example Use Case
        A hedge fund manager can use the Positions Summary API to track institutional ownership trends in Apple (AAPL), monitoring how many institutions are increasing or reducing their positions, and assessing overall market sentiment.
        """
        return params

    @BaseProxy.endpoint(
        category="Form 13F",
        sub_category="Symbol",
        endpoint="institutional-ownership/industry-summary",
        name="Industry Performance Summary API",
        description=(
            "The Industry Performance Summary API provides an overview of how various industries are performing financially. By analyzing the value of industries over a specific period, this API helps investors and analysts understand the health of entire sectors and make informed decisions about sector-based investments."
        ),
        params={"year*": (str, "2022"), "quarter*": (str, "3")},
        response=[
            {
                "industryTitle": "ABRASIVE, ASBESTOS & MISC NONMETALLIC MINERAL PRODS",
                "industryValue": 10979226300,
                "date": "2022-09-30",
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def industry_performance_summary(self, params: dict) -> dict:
        """
        About Industry Performance Summary API
        The Industry Performance Summary API enables users to retrieve financial performance summaries for specific industries. This API is ideal for:

        - Sector Analysis: Gain insights into how industries are performing, helping you identify strong or underperforming sectors.
        - Comparative Industry Health: Compare the financial health of different industries to assess which sectors might present better investment opportunities.
        - Macro-Level Market Insights: Use industry-level performance data to make informed decisions about broad market trends and economic shifts.
        This API offers a macroeconomic view of sector performance, making it a valuable tool for financial analysts, investors, and economists looking to understand industry-specific trends. It is a key tool for understanding industry trends and comparing the financial health of various sectors in the market.
        """
        return params

    ########################################
    ### Indexes Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Indexes",
        endpoint="index-list",
        name="Stock Market Indexes List API",
        description=(
            "Retrieve a comprehensive list of stock market indexes across global exchanges using the FMP Stock Market Indexes List API. This API provides essential information such as the symbol, name, exchange, and currency for each index, helping analysts and investors keep track of various market benchmarks."
        ),
        params={},
        response=[
            {
                "symbol": "^TTIN",
                "name": "S&P/TSX Capped Industrials Index",
                "exchange": "TSX",
                "currency": "CAD",
            },
        ],
    )
    def index_list(self, params: dict) -> dict:
        """
        About Stock Market Indexes List API
        The FMP Stock Market Indexes List API allows users to access a full directory of stock market indexes from exchanges worldwide. It provides detailed information about index symbols, names, exchanges, and currencies, making it a valuable resource for tracking market performance across different regions and sectors. Key features include:

        - Comprehensive Index Coverage: Access a wide range of indexes from major exchanges like NYSE, NASDAQ, and TSX.
        - Global Reach: The API offers data on indexes from international markets, providing a truly global perspective.
        - Basic Information on Each Index: Retrieve essential details such as the symbol, full name, and exchange, helping you identify the indexes relevant to your needs.
        - Currency Information: Understand the currency in which each index is denominated, enabling more accurate analysis for global investors.
        This API is particularly useful for investors, analysts, and portfolio managers who need to monitor market movements across multiple regions and sectors.

        Example Use Case
        A portfolio manager building a global investment strategy can use the Stock Market Indexes List API to retrieve data on key indexes from exchanges around the world. By identifying relevant indexes in different regions, they can assess market performance and make informed decisions about asset allocation.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="End Of Day",
        endpoint="historical-price-eod/light",
        name="Historical Stock Price Data API",
        description=(
            "Retrieve end-of-day historical prices for stock indexes using the Historical Price Data API. This API provides essential data such as date, price, and volume, enabling detailed analysis of price movements over time."
        ),
        params={
            "symbol*": (str, "^GSPC"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "^GSPC",
                "date": "2022-02-04",
                "price": 6037.89,
                "volume": 3020009000,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_data_light(self, params: dict) -> dict:
        """
        About Historical Stock Price Data API
        The FMP Historical Price Data API allows users to access end-of-day price data for stock indexes, offering insights into historical performance. By tracking this data, analysts can better understand market trends, volatility, and stock index movements. Key features include:

        - Comprehensive Price Data: Retrieve historical prices for key stock indexes, including data on closing price, date, and trading volume.
        - Supports Multiple Indexes: Access data for a wide range of stock indexes from various global markets.
        - Detailed Volume Information: Track trading volume for each index, offering insights into market activity levels.
        - Historical Performance Analysis: Analyze past price movements to identify trends, patterns, and potential investment opportunities.
        This API is particularly useful for financial analysts, investors, and market researchers who need accurate historical data to assess stock index performance over time.

        Example Use Case
        An investment analyst is developing a historical trend analysis for the S&P 500 index (^GSPC). By using the Historical Price Data API, they can retrieve end-of-day prices for specific dates, analyze the volume and price movements over time, and present findings to their clients for more informed investment decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="End Of Day",
        endpoint="historical-price-eod/full",
        name="Detailed Historical Stock Price Data API",
        description=(
            "Access full historical end-of-day prices for stock indexes using the Detailed Historical Price Data API. This API provides comprehensive information, including open, high, low, close prices, volume, and additional metrics for detailed financial analysis."
        ),
        params={
            "symbol*": (str, "^GSPC"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "^GSPC",
                "date": "2022-02-04",
                "open": 5998.14,
                "high": 6042.48,
                "low": 5990.87,
                "close": 6037.89,
                "volume": 3020009000,
                "change": 39.75,
                "changePercent": 0.66271,
                "vwap": 6017.345,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_price_data_full(self, params: dict) -> dict:
        """
        About Detailed Historical Stock Price Data API
        The FMP Detailed Historical Price Data API offers full end-of-day price data for stock indexes, making it a powerful tool for in-depth financial analysis. It includes a range of price points—open, high, low, close—along with volume, price changes, and volume-weighted average price (VWAP). Key features include:

        - Complete Price Data: Access open, high, low, and close prices for stock indexes on specific dates.
        - Volume Information: Track trading volume to assess market activity and liquidity.
        - Price Movement Insights: Analyze daily price changes and percentage changes to understand market trends.
        - Volume-Weighted Average Price (VWAP): Get VWAP data for each trading day, helping in performance benchmarking and trading decisions.
        This API is ideal for financial analysts, quants, and traders who need comprehensive historical price data to build models, conduct backtesting, or analyze market trends.

        Example Use Case
        A quantitative analyst developing an algorithmic trading model requires complete historical price data for the S&P 500 index (^GSPC). Using the Detailed Historical Price Data API, they can retrieve open, high, low, and close prices, along with VWAP and volume data for each trading day. This detailed information helps refine the model’s predictions and backtesting performance.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Intraday",
        endpoint="historical-chart/1min",
        name="1-Minute Interval Stock Price API",
        description="Retrieve 1-minute interval intraday data for stock indexes using the Intraday 1-Minute Price Data API. This API provides granular price information, helping users track short-term price movements and trading volume within each minute.",
        params={
            "symbol*": (str, "^GSPC"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 15:59:00",
                "open": 6040.47,
                "low": 6037.08,
                "high": 6041.71,
                "close": 6037.08,
                "volume": 70033000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_1min(self, params: dict) -> dict:
        """
        About 1-Minute Interval Stock Price API
        The FMP Intraday 1-Minute Price Data API delivers high-frequency price data for stock indexes, offering insights into market fluctuations on a minute-by-minute basis. This level of detail is ideal for active traders and analysts who require real-time market insights for rapid decision-making. Key features include:

        - Granular Price Data: Access open, high, low, and close prices for each minute of the trading day.
        - Minute-by-Minute Tracking: Monitor short-term price movements and trends in real time.
        - Volume Information: Analyze trading volume for each minute, offering insights into market liquidity and activity levels.
        - Supports Intraday Trading: Perfect for day traders and high-frequency trading strategies that rely on detailed intraday data.
        This API is particularly useful for day traders, quants, and financial analysts who need real-time data to track rapid price movements and make timely trading decisions.

        Example Use Case
        A day trader specializing in short-term stock index trades uses the Intraday 1-Minute Price Data API to track real-time price changes in the S&P 500 index (^GSPC). With access to minute-by-minute data, they can react to price movements and adjust their trading strategies in real time, optimizing their entry and exit points for maximum profitability.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Intraday",
        endpoint="historical-chart/5min",
        name="5-Minute Interval Stock Price API",
        description="Retrieve 5-minute interval intraday price data for stock indexes using the Intraday 5-Minute Price Data API. This API provides crucial insights into price movements and trading volume within 5-minute windows, ideal for traders who require short-term data.",
        params={
            "symbol*": (str, "^GSPC"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 15:55:00",
                "open": 6038.16,
                "low": 6037.02,
                "high": 6041.71,
                "close": 6037.08,
                "volume": 179921000,
            }
        ],
    )
    def historical_chart_5min(self, params: dict) -> dict:
        """
        About 5-Minute Interval Stock Price API
        The FMP Intraday 5-Minute Price Data API offers real-time price and volume data for stock indexes, updated every 5 minutes during active market hours. This API is designed for traders and analysts who need detailed, short-term data to track price fluctuations and make timely decisions. Key features include:

        - 5-Minute Interval Data: Access open, high, low, and close prices for each 5-minute interval throughout the trading day.
        - Real-Time Tracking: Stay up-to-date with price changes and market trends in near real-time.
        - Volume Data: Analyze trading volume in 5-minute intervals to gauge market activity and liquidity.
        - Supports Short-Term Trading: Ideal for short-term and swing traders looking for frequent updates to inform their strategies.
        This API is perfect for day traders, quants, and financial professionals who need to monitor price movements closely and execute trades based on short-term fluctuations.

        Example Use Case
        A swing trader monitoring the S&P 500 index (^GSPC) uses the Intraday 5-Minute Price Data API to track price movements over the course of the trading day. By analyzing the 5-minute intervals, they can time their trades more accurately, reacting quickly to short-term market changes and optimizing their strategy for maximum return.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Intraday",
        endpoint="historical-chart/1hour",
        name="1-Hour Interval Stock Price API",
        description="Access 1-hour interval intraday data for stock indexes using the Intraday 1-Hour Price Data API. This API provides detailed price movements and volume within hourly intervals, making it ideal for tracking medium-term market trends during the trading day.",
        params={
            "symbol*": (str, "^GSPC"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 15:30:00",
                "open": 6030.14,
                "low": 6030.14,
                "high": 6041.71,
                "close": 6037.88,
                "volume": 930623000,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def historical_chart_1hour(self, params: dict) -> dict:
        """
        About 1-Hour Interval Stock Price API
        The FMP Intraday 1-Hour Price Data API delivers hourly price data for stock indexes, allowing analysts and traders to track market trends and price movements throughout the day. With open, high, low, and close prices for each hour, this API is suited for those monitoring medium-term intraday performance. Key features include:

        - Hourly Interval Data: Retrieve open, high, low, and close prices for stock indexes at 1-hour intervals throughout the trading day.
        - Track Medium-Term Movements: Perfect for traders and analysts interested in observing trends within hourly windows rather than minute-by-minute fluctuations.
        - Volume Data: Analyze hourly trading volumes to gain insights into market activity and liquidity.
        - Intraday Trading Support: Ideal for swing traders and medium-term strategies that require detailed data without overwhelming granularity.
        This API is particularly useful for traders, analysts, and portfolio managers who need to assess market behavior within hourly intervals to inform their trading decisions.

        Example Use Case
        A swing trader using the Intraday 1-Hour Price Data API monitors the S&P 500 index (^GSPC) to observe price movements across several trading hours. With hourly updates, they can identify emerging trends and adjust their positions without the need to track minute-by-minute fluctuations.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Historical Constituents",
        endpoint="historical-sp500-constituent",
        name="Historical S&P 500 API",
        description="Retrieve historical data for the S&P 500 index using the Historical S&P 500 API. Analyze past changes in the index, including additions and removals of companies, to understand trends and performance over time.",
        params={},
        response=[
            {
                "dateAdded": "December 23, 2022",
                "addedSecurity": "Workday, Inc.",
                "removedTicker": "AMTM",
                "removedSecurity": "Amentum",
                "date": "2022-12-22",
                "symbol": "WDAY",
                "reason": "Market capitalization change.",
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_sp500_constituent(self, params: dict) -> dict:
        """
        About Historical S&P 500 API
        The FMP Historical S&P 500 API provides comprehensive historical data on changes to the S&P 500 index. This includes information on when companies were added or removed, along with the reasons behind these changes. It is an essential tool for analysts, portfolio managers, and researchers who need to track historical performance and trends within this key stock index. Key features include:

        - Additions & Removals: Access historical records of companies added to or removed from the S&P 500, including relevant dates and reasons for the changes.
        - Market Capitalization Changes: Track changes in the index composition driven by shifts in market capitalization.
        - Historical Index Insights: Analyze how the composition of the S&P 500 has evolved over time and how these changes impact market performance.
        - Company-Specific Data: Retrieve details about each company that has been added or removed, including symbols and company names.
        This API is particularly useful for financial analysts, researchers, and portfolio managers who want to analyze how changes in the S&P 500 index affect long-term market trends.

        Example Use Case
        A financial researcher uses the Historical S&P 500 API to study how the composition of the index has changed over the last decade. By analyzing additions and removals, such as the recent inclusion of Dell Technologies (DELL) in place of Etsy (ETSY), they can assess how shifts in market capitalization and industry representation affect overall index performance.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Historical Constituents",
        endpoint="historical-nasdaq-constituent",
        name="Historical Nasdaq API",
        description="Access historical data for the Nasdaq index using the Historical Nasdaq API. Analyze changes in the index composition and view how it has evolved over time, including company additions and removals.",
        params={},
        response=[
            {
                "dateAdded": "December 23, 2022",
                "addedSecurity": "Axon Enterprise Inc.",
                "removedTicker": "SMCI",
                "removedSecurity": "Super Micro Computer Inc",
                "date": "2022-12-22",
                "symbol": "AXON",
                "reason": "Annual Re-ranking",
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_nasdaq_constituent(self, params: dict) -> dict:
        """
        About Historical Nasdaq API
        The FMP Historical Nasdaq API provides detailed historical records of changes to the Nasdaq index. This includes data on when companies were added or removed, along with reasons for these changes, such as re-rankings or market capitalization adjustments. It’s an essential tool for analysts and investors who want to track the Nasdaq’s historical performance and composition. Key features include:

        - Company Additions & Removals: Access historical data on which companies have been added or removed from the Nasdaq, including relevant dates.
        - Reasons for Changes: Understand why changes occurred in the index, such as re-rankings or shifts in market capitalization.
        - Historical Analysis: Analyze the evolution of the Nasdaq index composition over time and how it has impacted overall market performance.
        - Detailed Company Data: Retrieve information on specific companies added to or removed from the Nasdaq, including their symbol, name, and sector.
        This API is particularly useful for investors, analysts, and researchers who need to study historical trends and changes in the Nasdaq index.

        Example Use Case
        A market analyst uses the Historical Nasdaq API to study changes in the composition of the Nasdaq index over the last five years. By examining data like the inclusion of Arm Holdings (ARM) and the removal of Sirius XM (SIRI) in 2024, they can assess how industry shifts and market dynamics have influenced the index’s overall performance.
        """
        return params

    @BaseProxy.endpoint(
        category="Indexes",
        sub_category="Historical Constituents",
        endpoint="historical-dowjones-constituent",
        name="Historical Dow Jones API",
        description="Access historical data for the Dow Jones Industrial Average using the Historical Dow Jones API. Analyze changes in the index’s composition and study its performance across different periods.",
        params={},
        response=[
            {
                "dateAdded": "November 8, 2022",
                "addedSecurity": "Nvidia",
                "removedTicker": "INTC",
                "removedSecurity": "Intel Corporation",
                "date": "2022-11-07",
                "symbol": "NVDA",
                "reason": "Market capitalization change",
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_dowjones_constituent(self, params: dict) -> dict:
        """
        About Historical Dow Jones API
        The FMP Historical Dow Jones API offers detailed records of changes to the Dow Jones Industrial Average, one of the most widely recognized stock indexes in the world. This API allows users to access information on companies added or removed from the index, along with reasons for those changes. It’s an invaluable tool for anyone conducting historical analysis of this major market indicator. Key features include:

        - Company Additions & Removals: Access detailed data on which companies were added or removed from the Dow Jones index, including relevant dates.
        - Reasons for Changes: Understand why companies were added or removed, such as market capitalization shifts or industry reclassifications.
        - Historical Composition: Analyze how the makeup of the Dow Jones has changed over time and how these changes have impacted the overall index.
        - Detailed Company Data: Retrieve information on specific companies, including their symbols, names, and the date they were added or removed from the index.
        This API is ideal for investors, market analysts, and researchers who want to explore historical changes in the Dow Jones and understand the factors driving those changes.

        Example Use Case
        A market researcher uses the Historical Dow Jones API to study how the index has evolved over the past decade. By examining changes like the inclusion of Amazon (AMZN) and the removal of Walgreens Boots Alliance (WBA) in 2024, the researcher can better understand how shifts in market capitalization and industry performance have influenced the Dow Jones over time.
        """
        return params

    ########################################
    ### Insider Trades Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Latest",
        endpoint="insider-trading/latest",
        name="Latest Insider Trades API",
        description=(
            "Access the latest insider trading activity using the Latest Insider Trading API. Track which company insiders are buying or selling stocks and analyze their transactions."
        ),
        params={"date": (str, "2022-02-04"), "page": (int, 0), "limit": (int, 100)},
        response=[
            {
                "symbol": "APA",
                "filingDate": "2022-02-04",
                "transactionDate": "2022-02-01",
                "reportingCik": "0001380034",
                "companyCik": "0001841666",
                "transactionType": "M-Exempt",
                "securitiesOwned": 104398,
                "reportingName": "Hoyt Rebecca A",
                "typeOfOwner": "officer: Sr. VP, Chief Acct Officer",
                "acquisitionOrDisposition": "A",
                "directOrIndirect": "D",
                "formType": "4",
                "securitiesTransacted": 3450,
                "price": 0,
                "securityName": "Common Stock",
                "url": "https://www.sec.gov/Archives/edgar/data/1841666/000194906025000035/0001949060-25-000035-index.htm",
            }
        ],
        dt_cutoff=("filingDate", "%Y-%m-%d"),
    )
    def latest_insider_trading(self, params: dict) -> dict:
        """
        About Latest Insider Trading API
        The FMP Latest Insider Trading API provides up-to-date information on insider trading activities. This API enables users to track recent stock purchases and sales by company insiders, including directors and executives. With details on transaction dates, types, and amounts, this API offers insights into corporate behavior and potential market trends. Key features include:

        - Recent Insider Transactions: Access the most recent stock purchases or sales by company insiders.
        - Transaction Details: Retrieve detailed information about the type of transaction, the number of shares transacted, and the price.
        - Insider Roles: Identify the roles of the individuals involved in the transactions, such as directors or executives.
        - Comprehensive Data: Access key information such as filing date, transaction date, type of ownership, and links to official filings.
        This API is ideal for investors, analysts, and financial researchers who want to track insider trading activity to assess market sentiment or potential investment opportunities.

        Example Use Case
        A hedge fund manager uses the Latest Insider Trading API to monitor recent stock purchases by company directors. By analyzing a purchase made by Larry Glasscock (director of SPG), they can assess whether the insider's buying activity signals confidence in the company’s future performance and adjust their investment strategy accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Search",
        endpoint="insider-trading/search",
        name="Search Insider Trades API",
        description=(
            "Search insider trading activity by company or symbol using the Search Insider Trades API. Find specific trades made by corporate insiders, including executives and directors."
        ),
        params={
            "symbol*": (str, "AAPL"),
            "page": (int, 0),
            "limit": (int, 100),
            "reportingCik": (str, "0001496686"),
            "companyCik": (str, "0000320193"),
            "transactionType": (str, "S-Sale"),
        },
        response=[
            {
                "symbol": "AAPL",
                "filingDate": "2022-02-04",
                "transactionDate": "2022-02-03",
                "reportingCik": "0001214128",
                "companyCik": "0000320193",
                "transactionType": "S-Sale",
                "securitiesOwned": 4159576,
                "reportingName": "LEVINSON ARTHUR D",
                "typeOfOwner": "director",
                "acquisitionOrDisposition": "D",
                "directOrIndirect": "D",
                "formType": "4",
                "securitiesTransacted": 1516,
                "price": 226.3501,
                "securityName": "Common Stock",
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000019/0000320193-25-000019-index.htm",
            },
        ],
        dt_cutoff=("filingDate", "%Y-%m-%d"),
    )
    def search_insider_trades(self, params: dict) -> dict:
        """
        About Search Insider Trades API
        The FMP Search Insider Trades API allows users to search for specific insider trading activities based on a company or stock symbol. This API provides detailed information on stock transactions by corporate insiders, including transaction dates, types, amounts, and roles within the company. Key features include:

        - Company-Specific Searches: Search insider trading activity by entering the stock symbol or company name to retrieve relevant transactions.
        - Detailed Transaction Information: Access detailed data such as transaction type (purchase or sale), number of securities transacted, and price.
        - Insider Roles: Understand the roles of the insiders involved in the transactions, such as directors or executives.
        - Direct Links to Filings: Each transaction includes a link to the official SEC filing for deeper analysis and verification.
        This API is perfect for investors, financial researchers, and analysts who need to investigate insider trading activities of specific companies or individuals.

        Example Use Case
        An investment analyst uses the Search Insider Trades API to investigate recent sales of Apple (AAPL) stock by Chris Kondo, the Principal Accounting Officer. By retrieving detailed information about the transaction, including the sale of 8,706 shares at $225, the analyst can better assess the implications for the company’s financial performance and strategy.
        """
        return params

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Search",
        endpoint="insider-trading/reporting-name",
        name="Search Insider Trades by Reporting Name API",
        description=(
            "Search for insider trading activity by reporting name using the Search Insider Trades by Reporting Name API. Track trading activities of specific individuals or groups involved in corporate insider transactions."
        ),
        params={"name*": (str, "Zuckerberg")},
        response=[{"reportingCik": "0001548760", "reportingName": "Zuckerberg Mark"}],
    )
    def search_insider_trades_by_name(self, params: dict) -> dict:
        """
        About Search Insider Trades by Reporting Name API
        The FMP Search Insider Trades by Reporting Name API allows users to search for insider trading activities based on the name of a specific individual or group. This API provides key information such as the reporting CIK (Central Index Key) and the individual’s name associated with insider transactions, enabling users to monitor the trading activity of high-profile individuals or corporate executives. Key features include:

        - Name-Specific Searches: Easily search for insider trades by entering the name of a specific individual or entity.
        - Reporting CIK Information: Retrieve the reporting CIK for more in-depth tracking of insider activity across filings.
        - Track High-Profile Insiders: Monitor trades by well-known corporate executives, directors, or other insiders.
        - Direct Access to Relevant Data: Quickly find information related to specific individuals’ insider trading activities, with links to more detailed data.
        This API is ideal for investors, analysts, and financial researchers who want to track insider trading activities associated with specific people or entities.

        Example Use Case
        A financial analyst uses the Search Insider Trades by Reporting Name API to track insider trading activity for Mark Zuckerberg. By retrieving the reporting CIK and related transactions, the analyst can monitor Zuckerberg’s trading behavior and analyze how his actions might influence market sentiment regarding Meta Platforms.
        """
        return params

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Statistics",
        endpoint="insider-trading-transaction-type",
        name="All Insider Transaction Types API",
        description=(
            "Access a comprehensive list of insider transaction types with the All Insider Transaction Types API. This API provides details on various transaction actions, including purchases, sales, and other corporate actions involving insider trading."
        ),
        params={},
        response=[
            {"transactionType": "A-Award"},
        ],
    )
    def insider_transaction_types(self, params: dict) -> dict:
        """
        About All Insider Transaction Types API
        The FMP All Insider Transaction Types API allows users to view all types of transactions made by corporate insiders. This includes purchases, sales, and other actions that insiders may take, such as options exercises or gifts. With this API, users can gain a comprehensive understanding of the different types of transactions insiders are reporting and their implications for company performance. Key features include:

        - Comprehensive Transaction Coverage: View all types of insider transactions, including buying, selling, option exercises, and more.
        - Transaction Classifications: Understand the classification of transactions, whether it's an acquisition, disposition, or other.
        - Real-Time Insights: Stay updated on the latest insider actions and their potential impact on the company.
        - Corporate Action Types: Access details on less common insider transactions, such as gifts or stock awards.
        This API is perfect for investors, analysts, and researchers who need to track a variety of insider trading actions to make more informed investment decisions.

        Example Use Case
        A market analyst uses the All Insider Transaction Types API to view a complete list of recent transactions by corporate insiders. By reviewing purchases, sales, and stock options exercised, the analyst can gain insights into corporate sentiment and make better-informed trading decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Statistics",
        endpoint="insider-trading/statistics",
        name="Insider Trade Statistics API",
        description=(
            "Analyze insider trading activity with the Insider Trade Statistics API. This API provides key statistics on insider transactions, including total purchases, sales, and trends for specific companies or stock symbols."
        ),
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "year": 2022,
                "quarter": 4,
                "acquiredTransactions": 6,
                "disposedTransactions": 38,
                "acquiredDisposedRatio": 0.1579,
                "totalAcquired": 994544,
                "totalDisposed": 2297088,
                "averageAcquired": 165757.3333,
                "averageDisposed": 60449.6842,
                "totalPurchases": 0,
                "totalSales": 22,
            },
        ],
        dt_cutoff=("year", "%Y"),
    )
    def insider_trade_statistics(self, params: dict) -> dict:
        """
        About Insider Trade Statistics API
        The FMP Insider Trade Statistics API provides comprehensive statistical data on insider trading activity for a specific stock symbol. This includes the total number of transactions, shares acquired or disposed of, and the overall ratio of acquisitions to dispositions. By analyzing these trends, users can gain insights into corporate sentiment and market behavior. Key features include:

        - Transaction Breakdown: Access statistics on insider acquisitions and dispositions for a specific company.
        - Acquired vs. Disposed Ratio: Analyze the ratio of shares acquired to shares disposed of, revealing insider sentiment.
        - Quarterly Data: View insider trading activity on a quarterly basis, helping you track changes in trading patterns over time.
        - Total and Average Transactions: Get detailed statistics on total purchases and sales, along with average transaction sizes.
        This API is ideal for investors, analysts, and financial researchers who need to analyze patterns and trends in insider trading activity to make informed investment decisions.

        Example Use Case
        A financial analyst uses the Insider Trade Statistics API to examine insider trading trends for Apple (AAPL) in the third quarter of 2024. By reviewing the ratio of shares disposed of to those acquired, along with the total number of sales, the analyst can assess whether insiders are showing confidence in the company’s future.
        """
        return params

    @BaseProxy.endpoint(
        category="Insider Trades",
        sub_category="Acquisition Ownership",
        endpoint="acquisition-of-beneficial-ownership",
        name="Acquisition Ownership API",
        description=(
            "Track changes in stock ownership during acquisitions using the Acquisition Ownership API. This API provides detailed information on how mergers, takeovers, or beneficial ownership changes impact the stock ownership structure of a company."
        ),
        params={"symbol*": (str, "AAPL"), "limit": (int, 2000)},
        response=[
            {
                "cik": "0000320193",
                "symbol": "AAPL",
                "filingDate": "2022-02-14",
                "acceptedDate": "2022-02-14",
                "cusip": "037833100",
                "nameOfReportingPerson": "National Indemnity Company",
                "citizenshipOrPlaceOfOrganization": "State of Nebraska",
                "soleVotingPower": "0",
                "sharedVotingPower": "755059877",
                "soleDispositivePower": "0",
                "sharedDispositivePower": "755059877",
                "amountBeneficiallyOwned": "755059877",
                "percentOfClass": "4.8",
                "typeOfReportingPerson": "IC, EP, IN, CO",
                "url": "https://www.sec.gov/Archives/edgar/data/320193/000119312524036431/d751537dsc13ga.htm",
            },
        ],
        dt_cutoff=("filingDate", "%Y-%m-%d"),
    )
    def acquisition_ownership(self, params: dict) -> dict:
        """
        About Acquisition Ownership API
        The FMP Acquisition Ownership API provides comprehensive data on changes in stock ownership during acquisitions, mergers, or other significant corporate events. It offers insight into how control and ownership are transferred or shared between entities, helping analysts and investors understand the impact of these changes on corporate governance and shareholder influence. Key features include:

        - Ownership Changes: Track changes in beneficial ownership, including shared or sole voting and dispositive powers.
        - Acquisition and Merger Data: View details about mergers, takeovers, or acquisitions that affect the ownership of company stock.
        - Detailed Reporting Information: Access data about the reporting entities, including their CIK, name, and percentage of ownership.
        - Filing Dates and SEC Links: Get links to official SEC filings and important dates related to acquisitions or ownership changes.
        This API is ideal for investors, financial analysts, and researchers who need to track how ownership structures shift during corporate acquisitions or mergers.

        Example Use Case
        An institutional investor uses the Acquisition Ownership API to monitor the impact of a recent merger involving Apple (AAPL). By examining the beneficial ownership change reported by National Indemnity Company, which now holds 755 million shares, the investor can assess how this affects voting power and control within the company.
        """
        return params

    ########################################
    ### Market Performance Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="Market Performance",
        endpoint="sector-performance-snapshot",
        name="Market Sector Performance Snapshot API",
        description=(
            "Get a snapshot of sector performance using the Market Sector Performance Snapshot API. Analyze how different industries are performing in the market based on average changes across sectors."
        ),
        params={
            "date*": (str, "2022-02-01"),
            "exchange": (str, "NASDAQ"),
            "sector": (str, "Basic Materials"),
        },
        response=[
            {
                "date": "2022-02-01",
                "sector": "Basic Materials",
                "exchange": "NASDAQ",
                "averageChange": -0.31481377464310634,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def sector_performance_snapshot(self, params: dict) -> dict:
        """
        About Market Sector Performance Snapshot API
        The FMP Market Sector Performance Snapshot API provides real-time insights into the performance of different sectors across various stock exchanges. This API allows users to track the average performance of industries like Basic Materials, Technology, Healthcare, and more, helping analysts and investors understand how different parts of the market are doing at any given moment. Key features include:

        - Sector-Specific Performance Data: Access performance data for various sectors, including the average percentage change for each sector.
        - Exchange-Specific Analysis: Analyze sector performance across specific exchanges such as NASDAQ, NYSE, and others.
        - Daily Snapshots: Get daily updates on sector performance to track trends and market dynamics in real time.
        - Cross-Industry Comparisons: Compare the performance of different sectors to identify growth or decline in key areas of the market.
        This API is ideal for financial analysts, portfolio managers, and traders who need to track sector-level performance to make informed investment decisions.

        Example Use Case
        A portfolio manager uses the Market Sector Performance Snapshot API to review how different sectors performed on NASDAQ on a specific date. By identifying that the Basic Materials sector experienced an average decline of -0.31%, the manager can adjust their sector allocations and shift their focus to outperforming industries.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="Market Performance",
        endpoint="industry-performance-snapshot",
        name="Industry Performance Snapshot API",
        description=(
            "Access detailed performance data by industry using the Industry Performance Snapshot API. Analyze trends, movements, and daily performance metrics for specific industries across various stock exchanges."
        ),
        params={
            "date*": (str, "2022-02-01"),
            "exchange": (str, "NASDAQ"),
            "industry": (str, "Advertising Agencies"),
        },
        response=[
            {
                "date": "2022-02-01",
                "industry": "Advertising Agencies",
                "exchange": "NASDAQ",
                "averageChange": 3.8660194344955996,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def industry_performance_snapshot(self, params: dict) -> dict:
        """
        About Industry Performance Snapshot API
        The FMP Industry Performance Snapshot API provides a daily overview of how specific industries are performing across major stock exchanges. This API delivers key data, such as average percentage changes for industries like Advertising Agencies, Healthcare Equipment, or Technology Services, allowing users to track and compare performance trends within specific sectors. Key features include:

        - Industry-Level Performance Data: View average percentage changes for specific industries across major exchanges.
        - Real-Time Market Insights: Analyze industry performance trends and movements in real time with daily updates.
        - Exchange-Specific Data: Compare how different industries are performing on various stock exchanges like NASDAQ, NYSE, and others.
        - In-Depth Industry Comparisons: Track and analyze the performance of specific industries to understand market trends and identify growth opportunities.
        This API is ideal for market analysts, portfolio managers, and investors who need to understand the performance dynamics of individual industries to guide investment strategies.

        Example Use Case
        A market analyst uses the Industry Performance Snapshot API to analyze the performance of the Advertising Agencies industry on a specific date, and finds that it posted an average gain of 3.87% on NASDAQ. This data helps the analyst recommend sector-specific investments and identify growth trends in the advertising sector.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="Market Performance",
        endpoint="historical-sector-performance",
        name="Historical Market Sector Performance API",
        description=(
            "Access historical sector performance data using the Historical Market Sector Performance API. Review how different sectors have performed over time across various stock exchanges."
        ),
        params={
            "sector*": (str, "Energy"),
            "exchange": (str, "NASDAQ"),
            "from": (str, "2022-02-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-01",
                "sector": "Energy",
                "exchange": "NASDAQ",
                "averageChange": 0.6397534025664513,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_sector_performance(self, params: dict) -> dict:
        """
        About Historical Market Sector Performance API
        The FMP Historical Market Sector Performance API provides detailed historical data on the performance of market sectors, such as Energy, Technology, Healthcare, and others. This API allows users to track and analyze sector-specific trends over time, helping identify long-term patterns and market movements. Key features include:

        - Historical Sector Performance: Access historical data on average percentage changes in various sectors over time.
        - Exchange-Specific Data: Track how sectors have performed on different stock exchanges, including NASDAQ, NYSE, and others.
        - Long-Term Market Trends: Analyze trends and sector performance data over extended periods, offering insights for long-term investment strategies.
        - Cross-Sector Analysis: Compare the performance of multiple sectors to see how different areas of the market have evolved.
        This API is ideal for financial researchers, portfolio managers, and investors who need to review historical sector performance for trend analysis, sector rotation strategies, and long-term planning.

        Example Use Case
        An investor uses the Historical Market Sector Performance API to review the Energy sector’s historical performance on NASDAQ. By analyzing data from a specific date, showing an average change of 0.64%, the investor can track the sector's performance over time and make more informed decisions about future investments in the Energy sector.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="Market Performance",
        endpoint="historical-industry-performance",
        name="Historical Industry Performance API",
        description=(
            "Access historical performance data for industries using the Historical Industry Performance API. Track long-term trends and analyze how different industries have evolved over time across various stock exchanges."
        ),
        params={
            "industry*": (str, "Energy"),
            "exchange": (str, "NASDAQ"),
            "from": (str, "2022-02-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-01",
                "industry": "Biotechnology",
                "exchange": "NASDAQ",
                "averageChange": 1.1479066960358322,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_industry_performance(self, params: dict) -> dict:
        """
        About Historical Industry Performance API
        The FMP Historical Industry Performance API provides detailed historical data on the performance of various industries, such as Biotechnology, Technology, Financial Services, and more. This API allows users to track industry-specific performance metrics over time, providing insights into long-term trends and movements within the market. Key features include:

        - Industry-Level Historical Data: Access performance data for specific industries, including average percentage changes over time.
        - Exchange-Specific Performance: View how industries have performed on major stock exchanges like NASDAQ, NYSE, and others.
        - Long-Term Trend Analysis: Analyze historical data to identify long-term industry trends and market shifts.
        - Cross-Industry Comparisons: Compare the performance of different industries over time to identify growth areas and declining sectors.
        This API is ideal for market analysts, portfolio managers, and investors who need to track industry-level performance trends to guide long-term investment strategies.

        Example Use Case
        A financial analyst uses the Historical Industry Performance API to track the historical performance of the Biotechnology industry on NASDAQ. By reviewing data from a specific date, showing an average gain of 1.15%, the analyst can assess how the industry has performed over time and determine if it aligns with their investment strategy.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="PE Ratio",
        endpoint="sector-pe-snapshot",
        name="Sector PE Snapshot API",
        description=(
            "Retrieve the price-to-earnings (P/E) ratios for various sectors using the Sector P/E Snapshot API. Compare valuation levels across sectors to better understand market valuations."
        ),
        params={
            "date*": (str, "2022-02-01"),
            "exchange": (str, "NASDAQ"),
            "sector": (str, "Basic Materials"),
        },
        response=[
            {
                "date": "2022-02-01",
                "sector": "Basic Materials",
                "exchange": "NASDAQ",
                "pe": 15.687711758428254,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def sector_pe_snapshot(self, params: dict) -> dict:
        """
        About Sector PE Snapshot API
        The FMP Sector P/E Snapshot API provides detailed data on the price-to-earnings (P/E) ratios of different market sectors, such as Basic Materials, Technology, Healthcare, and more. This API allows users to analyze sector-specific valuations, providing insights into how sectors are valued relative to their earnings. Key features include:

        - P/E Ratio by Sector: Access up-to-date P/E ratios for various sectors, helping you compare their relative valuations.
        - Exchange-Specific Data: View sector P/E ratios for specific exchanges, such as NASDAQ or NYSE.
        - Daily Updates: Receive daily updates on sector P/E ratios to track changes in valuation levels over time.
        - Valuation Comparisons: Compare the P/E ratios across multiple sectors to identify potential overvalued or undervalued sectors.
        This API is ideal for investors, analysts, and portfolio managers who need to assess sector valuations for investment decision-making and market analysis.

        Example Use Case
        A portfolio manager uses the Sector P/E Snapshot API to compare the P/E ratios of different sectors on NASDAQ. By seeing that the Basic Materials sector has a P/E ratio of 15.69, they can assess whether this sector is overvalued or undervalued relative to other sectors and adjust their portfolio accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="PE Ratio",
        endpoint="industry-pe-snapshot",
        name="Industry PE Snapshot API",
        description=(
            "View price-to-earnings (P/E) ratios for different industries using the Industry P/E Snapshot API. Analyze valuation levels across various industries to understand how each is priced relative to its earnings."
        ),
        params={
            "date*": (str, "2022-02-01"),
            "exchange": (str, "NASDAQ"),
            "industry": (str, "Advertising Agencies"),
        },
        response=[
            {
                "date": "2022-02-01",
                "industry": "Advertising Agencies",
                "exchange": "NASDAQ",
                "pe": 71.09601665201151,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def industry_pe_snapshot(self, params: dict) -> dict:
        """
        About Industry PE Snapshot API
        The FMP Industry P/E Snapshot API provides detailed information on the price-to-earnings (P/E) ratios of various industries, such as Advertising Agencies, Technology, and Healthcare. This API enables users to compare industry-specific valuation levels across stock exchanges like NASDAQ and NYSE, offering insights into which industries are overvalued or undervalued. Key features include:

        - P/E Ratios by Industry: Access the most recent P/E ratios for industries across major stock exchanges.
        - Daily Updates: Get daily snapshots of industry P/E ratios, helping track changes in valuations over time.
        - Exchange-Specific Data: Analyze how industries are valued on different exchanges, such as NASDAQ or NYSE.
        - Cross-Industry Comparisons: Compare P/E ratios across industries to identify potential investment opportunities or risks.
        This API is perfect for investors, analysts, and financial professionals looking to evaluate industry-specific valuations for making informed investment decisions.

        Example Use Case
        An investor uses the Industry P/E Snapshot API to assess a specific industry on NASDAQ. Knowing the P/E ratio, the investor can determine if the industry is overvalued and adjust their portfolio accordingly.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="PE Ratio",
        endpoint="historical-sector-pe",
        name="Historical Sector PE API",
        description=(
            "Access historical price-to-earnings (P/E) ratios for various sectors using the Historical Sector P/E API. Analyze how sector valuations have evolved over time to understand long-term trends and market shifts."
        ),
        params={
            "sector*": (str, "Energy"),
            "exchange": (str, "NASDAQ"),
            "from": (str, "2022-02-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-01",
                "sector": "Energy",
                "exchange": "NASDAQ",
                "pe": 14.411400922841464,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_sector_pe(self, params: dict) -> dict:
        """
        About Historical Sector PE API
        The FMP Historical Sector P/E API provides detailed historical data on the price-to-earnings (P/E) ratios of different sectors, such as Energy, Technology, and Healthcare. This API helps users track how sector valuations have changed over time, offering insights into long-term trends and shifts in market sentiment. Key features include:

        - Historical P/E Ratios by Sector: Access historical P/E ratios for various sectors, allowing you to track valuation trends.
        - Exchange-Specific Data: Analyze sector valuations on specific exchanges, such as NASDAQ or NYSE.
        - Long-Term Analysis: Review historical data to identify sector trends and how valuations have evolved over time.
        - Cross-Sector Comparisons: Compare P/E ratios across multiple sectors to better understand relative valuations and market shifts.
        This API is ideal for market analysts, portfolio managers, and investors who need to analyze sector-level valuation trends for long-term investment strategies.

        Example Use Case
        A portfolio manager uses the Historical Sector P/E API to review the historical P/E ratios of the Energy sector on NASDAQ. By examining the changes in P/E ratios over time, the manager can assess how the sector's valuation has evolved and make informed decisions about future investments.
        """
        return params

    @BaseProxy.endpoint(
        category="Market Performance",
        sub_category="PE Ratio",
        endpoint="historical-industry-pe",
        name="Historical Industry PE API",
        description=(
            "Access historical price-to-earnings (P/E) ratios by industry using the Historical Industry P/E API. Track valuation trends across various industries to understand how market sentiment and valuations have evolved over time."
        ),
        params={
            "industry*": (str, "Biotechnology"),
            "exchange": (str, "NASDAQ"),
            "from": (str, "2022-02-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-01",
                "industry": "Biotechnology",
                "exchange": "NASDAQ",
                "pe": 10.181600321811821,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def historical_industry_pe(self, params: dict) -> dict:
        """
        About Historical Industry PE API
        The FMP Historical Industry P/E API provides detailed historical data on the price-to-earnings (P/E) ratios of different industries, such as Biotechnology, Financial Services, and Consumer Goods. This API helps users track how industry valuations have changed over time, offering insights into long-term trends and market shifts. Key features include:

        - Industry-Specific P/E Data: Access historical P/E ratios for specific industries, helping you track how valuations have evolved over time.
        - Exchange-Specific Analysis: View industry P/E ratios across different exchanges, including NASDAQ, NYSE, and more.
        - Long-Term Valuation Trends: Analyze historical data to identify valuation trends and shifts in market sentiment within industries.
        - Cross-Industry Comparisons: Compare P/E ratios across multiple industries to understand which sectors are undervalued or overvalued.
        This API is ideal for investors, market analysts, and portfolio managers who need to assess industry-specific valuation trends to inform long-term investment strategies.

        Example Use Case
        A financial analyst uses the Historical Industry P/E API to review the historical P/E ratios of the Biotechnology industry on NASDAQ. By tracking how the P/E ratio has evolved over time, the analyst can determine whether the industry’s current valuation reflects long-term market trends and decide if it's a good investment opportunity.
        """
        return params

    ########################################
    ### ETF & Mutual Funds Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Holdings",
        endpoint="etf/info",
        name="ETF & Mutual Fund Information API",
        description=(
            "Access comprehensive data on ETFs and mutual funds with the FMP ETF & Mutual Fund Information API. Retrieve essential details such as ticker symbol, fund name, expense ratio, assets under management, and more."
        ),
        remove_keys=[
            "assetsUnderManagement",
            "nav",
            "navCurrency",
            "updatedAt",
            "avgVolume",
        ],
        params={"symbol*": (str, "SPY")},
        response=[
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF Trust",
                "description": "The Trust seeks to achieve its investment objective by holding a portfolio of the common stocks that are included in the index (the “Portfolio”), with the weight of each stock in the Portfolio substantially corresponding to the weight of such stock in the index.",
                "isin": "US78462F1030",
                "assetClass": "Equity",
                "securityCusip": "78462F103",
                "domicile": "US",
                "website": "https://www.ssga.com/us/en/institutional/etfs/spdr-sp-500-etf-trust-spy",
                "etfCompany": "SPDR",
                "expenseRatio": 0.0945,
                "inceptionDate": "1993-01-22",
                "holdingsCount": 503,
                "sectorsList": [
                    {"industry": "Basic Materials", "exposure": 1.97},
                    {"industry": "Communication Services", "exposure": 8.87},
                    {"industry": "Consumer Cyclical", "exposure": 9.84},
                ],
            }
        ],
    )
    def etf_info(self, params: dict) -> dict:
        """
        About ETF & Mutual Fund Information API
        The FMP ETF & Mutual Fund Information API offers a detailed look into the financial and structural information of ETFs and mutual funds. This API enables investors to:

        - Compare Funds: Evaluate different ETFs and mutual funds by reviewing key metrics like ticker symbol, name, expense ratio, and assets under management to choose the most cost-effective and suitable investment options.
        - Identify Investment Opportunities: Use the detailed data to discover ETFs and mutual funds that align with your specific investment strategy, risk tolerance, and financial goals.
        - Understand Investment Objectives: Learn more about the objectives and strategies of various ETFs and mutual funds, helping you assess their suitability for inclusion in your portfolio based on asset class, sector exposure, and expense ratios.
        For example, an investor can use this API to compare the expense ratios of various ETFs and mutual funds, find funds with large assets under management, or analyze sector weightings to ensure their investments align with their market outlook.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Holdings",
        endpoint="etf/country-weightings",
        name="ETF & Fund Country Allocation API",
        description=(
            "Gain insight into how ETFs and mutual funds distribute assets across different countries with the FMP ETF & Fund Country Allocation API. This tool provides detailed information on the percentage of assets allocated to various regions, helping you make informed investment decisions."
        ),
        params={"symbol*": (str, "SPY")},
        response=[
            {"country": "United States", "weightPercentage": "97.29%"},
        ],
    )
    def etf_country_weightings(self, params: dict) -> dict:
        """
        About ETF & Fund Country Allocation API
        The FMP ETF & Fund Country Allocation API delivers a detailed breakdown of how ETFs and mutual funds allocate their assets by country. This data is essential for investors aiming to:

        - Assess Geographic Exposure: Understand how assets are distributed globally, offering insights into the geographic risk and opportunities associated with different funds.
        - Identify Country-Specific Investment Opportunities: Evaluate funds with significant exposure to countries that show strong economic growth potential, like the United States, China, or emerging markets.
        - Diversify Your Portfolio: Use country allocation data to balance your investments across international markets, reducing concentration risk in any single region.
        For example, if you're looking to invest in a fund that heavily allocates its assets to the United States, you can use this API to find ETFs or mutual funds with a high percentage of their holdings in the U.S. Alternatively, if you want to diversify into international markets, this API will help you locate funds with significant exposure to foreign economies.

        Example Use Case
        An investor seeking to minimize risk by diversifying internationally might use the ETF & Fund Country Allocation API to identify funds with strong exposure to emerging markets or regions like Asia or Europe.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Holdings",
        endpoint="etf/asset-exposure",
        name="ETF Asset Exposure API",
        description=(
            "Discover which ETFs hold specific stocks with the FMP ETF Asset Exposure API. Access detailed information on market value, share numbers, and weight percentages for assets within ETFs."
        ),
        remove_keys=["marketValue"],
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "ZECP",
                "asset": "AAPL",
                "sharesNumber": 5482,
                "weightPercentage": 5.86,
            },
        ],
    )
    def etf_asset_exposure(self, params: dict) -> dict:
        """
        About ETF Asset Exposure API
        The FMP ETF Asset Exposure API provides detailed data on the exposure of individual stocks within various ETFs. This API is essential for:

        - Identifying ETF Holdings: Find out which ETFs hold a particular stock, along with details such as market value, the number of shares held, and the weight percentage of the stock within the ETF.
        - Analyzing Asset Exposure: Use the data to analyze the exposure of specific assets within ETFs, helping you understand how widely a stock is held and its significance within different funds.
        - Informed Investment Decisions: Investors can leverage this information to assess the popularity and weight of a stock across multiple ETFs, guiding their decisions on buying or selling the stock based on its representation in the market.
        This API is a valuable resource for investors who want to explore the relationship between stocks and ETFs, particularly for understanding the broader market sentiment towards a specific asset.

        Example Use Cases
        ETF Research: An investor interested in Apple Inc. (AAPL) can use the ETF Asset Exposure API to find all ETFs that hold AAPL shares. The investor can then analyze the weight of AAPL within each ETF to determine which funds are most heavily invested in the stock.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Holding",
        endpoint="etf/sector-weightings",
        name="ETF Sector Weighting API",
        description=(
            "The FMP ETF Sector Weighting API provides a breakdown of the percentage of an ETF's assets that are invested in each sector. For example, an investor may want to invest in an ETF that has a high exposure to the technology sector if they believe that the technology sector is poised for growth."
        ),
        params={"symbol*": (str, "SPY")},
        response=[
            {"symbol": "SPY", "sector": "Basic Materials", "weightPercentage": 1.97},
        ],
    )
    def etf_sector_weightings(self, params: dict) -> dict:
        """
        The FMP ETF Asset Exposure API provides detailed data on the exposure of individual stocks within various ETFs. This API is essential for:

        - Identifying ETF Holdings: Find out which ETFs hold a particular stock, along with details such as market value, the number of shares held, and the weight percentage of the stock within the ETF.
        - Analyzing Asset Exposure: Use the data to analyze the exposure of specific assets within ETFs, helping you understand how widely a stock is held and its significance within different funds.
        - Informed Investment Decisions: Investors can leverage this information to assess the popularity and weight of a stock across multiple ETFs, guiding their decisions on buying or selling the stock based on its representation in the market.
        This API is a valuable resource for investors who want to explore the relationship between stocks and ETFs, particularly for understanding the broader market sentiment towards a specific asset.

        Example Use Cases
        ETF Research: An investor interested in Apple Inc. (AAPL) can use the ETF Asset Exposure API to find all ETFs that hold AAPL shares. The investor can then analyze the weight of AAPL within each ETF to determine which funds are most heavily invested in the stock.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Fund Disclosures",
        endpoint="funds/disclosure-holders-latest",
        name="Mutual Fund & ETF Disclosure API",
        description=(
            "Access the latest disclosures from mutual funds and ETFs with the FMP Mutual Fund & ETF Disclosure API. This API provides updates on filings, changes in holdings, and other critical disclosure data for mutual funds and ETFs."
        ),
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "cik": "0000106444",
                "holder": "VANGUARD FIXED INCOME SECURITIES FUNDS",
                "shares": 67030000,
                "dateReported": "2022-07-31",
                "change": 0,
                "weightPercent": 0.03840197,
            },
        ],
        dt_cutoff=("dateReported", "%Y-%m-%d"),
    )
    def funds_disclosure_holders_latest(self, params: dict) -> dict:
        """
        About Mutual Fund & ETF Disclosure API
        The FMP Mutual Fund & ETF Disclosure API delivers up-to-date information on the holdings and strategy changes of mutual funds and ETFs. This API is designed for investors, analysts, and financial professionals who need to:

        - Track Fund Holdings: Stay informed on the latest holdings disclosed by mutual funds and ETFs, including the number of shares held and the percentage of the portfolio they represent.
        - Monitor Strategy Changes: Detect changes in fund strategy by reviewing updated disclosures, which may reveal shifts in investment focus or portfolio rebalancing.
        - Gain Insight into Major Funds: Understand the investment decisions of significant institutional players, such as Vanguard or BlackRock, by accessing their most recent filings.
        For example, an investor might use this API to track the latest disclosure from Vanguard’s mutual fund, analyzing whether the fund increased or decreased its position in a particular stock, and use that information to support their own investment strategy.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Fund Disclosures",
        endpoint="funds/disclosure",
        name="Mutual Fund Disclosures API",
        description=(
            "Access comprehensive disclosure data for mutual funds with the FMP Mutual Fund Disclosures API. Analyze recent filings, balance sheets, and financial reports to gain insights into mutual fund portfolios."
        ),
        params={
            "symbol*": (str, "VWO"),
            "year*": (str, "2022"),
            "quarter*": (str, "4"),
            "cik": (str, "0000036405"),
        },
        response=[
            {
                "cik": "0000857489",
                "date": "2022-10-31",
                "acceptedDate": "2022-12-28 09:26:13",
                "symbol": "000089.SZ",
                "name": "Shenzhen Airport Co Ltd",
                "lei": "3003009W045RIKRBZI44",
                "title": "SHENZ AIRPORT-A",
                "cusip": "N/A",
                "isin": "CNE000000VK1",
                "balance": 2438784,
                "units": "NS",
                "cur_cd": "CNY",
                "valUsd": 2255873.6,
                "pctVal": 0.0023838966190458215,
                "payoffProfile": "Long",
                "assetCat": "EC",
                "issuerCat": "CORP",
                "invCountry": "CN",
                "isRestrictedSec": "N",
                "fairValLevel": "2",
                "isCashCollateral": "N",
                "isNonCashCollateral": "N",
                "isLoanByFund": "N",
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def funds_disclosure(self, params: dict) -> dict:
        """
        About Mutual Fund Disclosures API
        The FMP Mutual Fund Disclosures API provides detailed information on mutual fund holdings and recent filings, allowing investors and financial professionals to:

        - Track Fund Holdings: Review the most recent disclosures of mutual fund holdings, including asset categories, issuer information, and country of investment. This helps users understand the portfolio composition of various mutual funds.
        - Analyze Recent Filings: Obtain critical financial reports and filings from mutual funds, including balance data, market value in USD, percentage of total portfolio value, and more. These insights can help with investment analysis and strategy development.
        - Gain Transparency into Investments: The API provides essential details like CUSIP, ISIN, issuer category, and fair value levels, offering full transparency into mutual fund investments.
        For example, an investor can use this API to review the holdings of a mutual fund, such as Realty Income Corp, analyzing the balance, value in USD, and percentage of portfolio allocation to help make informed investment decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Funds",
        sub_category="Fund Disclosures",
        endpoint="funds/disclosure-holders-search",
        name="Mutual Fund & ETF Disclosure Name Search API",
        description=(
            "Easily search for mutual fund and ETF disclosures by name using the Mutual Fund & ETF Disclosure Name Search API. This API allows you to find specific reports and filings based on the fund or ETF name, providing essential details like CIK number, entity information, and reporting file number."
        ),
        params={"name*": (str, "Federated Hermes Government Income Securities, Inc.")},
        response=[
            {
                "symbol": "FGOAX",
                "cik": "0000355691",
                "classId": "C000024574",
                "seriesId": "S000009042",
                "entityName": "Federated Hermes Government Income Securities, Inc.",
                "entityOrgType": "30",
                "seriesName": "Federated Hermes Government Income Securities, Inc.",
                "className": "Class A Shares",
                "reportingFileNumber": "811-03266",
                "address": "4000 ERICSSON DRIVE",
                "city": "WARRENDALE",
                "zipCode": "15086-7561",
                "state": "PA",
            },
        ],
    )
    def funds_disclosure_holders_search(self, params: dict) -> dict:
        """
        About Mutual Fund & ETF Disclosure Name Search API
        The Mutual Fund & ETF Disclosure Name Search API helps users quickly locate disclosure documents for mutual funds and ETFs by searching with a specific fund name. It returns critical data such as the fund's symbol, CIK, class information, and the address of the reporting entity. Ideal for investors, analysts, and researchers looking for detailed disclosure information for compliance, research, or investment decision-making.

        - Fund Name Search: Look up disclosures for mutual funds and ETFs using the fund or entity name.
        - Key Filing Details: Get important information like CIK number, series and class IDs, entity name, and reporting file number.
        - Comprehensive Results: The API returns address details and filing information for the searched fund or ETF entity, making it easy to locate relevant documents.
        This API is perfect for anyone conducting due diligence or research on mutual funds and ETFs, allowing for precise and efficient disclosure searches.

        Example Use Case
        A financial analyst can use the Mutual Fund & ETF Disclosure Name Search API to retrieve specific disclosures for a mutual fund by entering its name, helping the analyst review relevant regulatory filings and reports for the fund.
        """
        return params

    @BaseProxy.endpoint(
        category="ETF & Mutual Fund",
        sub_category="Fund Disclosures",
        endpoint="funds/disclosure-dates",
        name="Fund & ETF Disclosures by Date API",
        description=(
            "Retrieve detailed disclosures for mutual funds and ETFs based on filing dates with the FMP Fund & ETF Disclosures by Date API. Stay current with the latest filings and track regulatory updates effectively."
        ),
        params={"symbol*": (str, "SPY"), "cik": (str, "0000036405")},
        response=[{"date": "2022-10-31", "year": 2022, "quarter": 4}],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def funds_disclosure_dates(self, params: dict) -> dict:
        """
        About Fund & ETF Disclosures by Date API
        The FMP Fund & ETF Disclosures by Date API allows users to quickly access mutual fund and ETF disclosures by specifying filing dates. This API is essential for:

        - Tracking Recent Filings: Stay informed about the latest mutual fund and ETF filings by retrieving disclosures based on specific filing dates. This feature is ideal for analysts, investors, and compliance officers looking to stay updated on current regulatory filings.
        - Historical Research: The API allows users to retrieve disclosures from past quarters or years, making it a valuable tool for historical financial research, performance tracking, and compliance verification.
        - Monitoring Filing Trends: Regularly reviewing filings by date helps users keep an eye on market trends and understand how recent filings may impact the financial markets.
        For example, an investor may want to track all disclosures filed in the second quarter of 2024. By using the Fund & ETF Disclosures by Date API, they can quickly retrieve and review these filings to understand any significant changes in fund strategies or holdings.
        """
        return params

    ########################################
    ### Commodity Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="Commodities",
        endpoint="commodities-list",
        name="Commodities List API",
        description=(
            "Access an extensive list of tracked commodities across various sectors, including energy, metals, and agricultural products. The FMP Commodities List API provides essential data on tradable commodities, giving investors the ability to explore market options in real-time."
        ),
        params={},
        response=[
            {
                "symbol": "HEUSX",
                "name": "Lean Hogs Futures",
                "exchange": None,
                "tradeMonth": "Dec",
                "currency": "USX",
            },
        ],
    )
    def commodities_list(self, params: dict) -> dict:
        """
        About Commodities List API
        The FMP Commodities List API offers users the ability to access a detailed list of tradable commodities. Whether you’re tracking energy futures, precious metals, or agricultural products, this API provides comprehensive data, including symbols, trade months, and associated currencies. Key features include:

        - Wide Commodity Coverage: View all available commodities across sectors such as energy (oil, natural gas), metals (gold, silver), and agriculture (corn, wheat). This diverse coverage makes it easier to find and analyze various markets.
        - Market Insights: With trade month and currency data provided, investors and analysts can better understand global market trends and pricing structures within the commodities sector.
        - Real-Time Data: Stay updated with the most current information on commodities, allowing for timely and informed investment decisions.
        For instance, users can access information on the "30 Day Fed Fund Futures" commodity, seeing details like its symbol, trade month, and associated currency, helping to track specific commodities for trading and hedging purposes.
        """
        return params

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="End Of Day",
        endpoint="historical-price-eod/light",
        name="Light Chart API",
        description=(
            "Access historical end-of-day prices for various commodities with the FMP Historical Commodities Price API. Analyze past price movements, trading volume, and trends to support informed decision-making."
        ),
        params={
            "symbol*": (str, "GCUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "GCUSD",
                "date": "2022-02-04",
                "price": 2873.7,
                "volume": 137844,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def commodities_light_chart(self, params: dict) -> dict:
        """
        About Light Chart API
        The FMP Historical Commodities Price API offers users access to end-of-day pricing data for a wide range of commodities. This API is designed for investors, traders, and analysts who need to perform historical analysis on commodities markets, track price trends, and make informed predictions based on past data.

        - End-of-Day Pricing: Retrieve accurate historical prices for commodities, including key metrics like trading volume, to analyze market performance over time.
        - Comprehensive Historical Data: Access a detailed record of price changes for commodities over any chosen period.
        - Trading Volume Insights: Evaluate the trading activity for each commodity with volume data included alongside price information.
        This API is ideal for financial professionals looking to analyze historical commodity data for research, risk management, or strategic trading purposes.
        """
        return params

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="End Of Day",
        endpoint="historical-price-eod/full",
        name="Full Chart API",
        description=(
            "Access full historical end-of-day price data for commodities with the FMP Comprehensive Commodities Price API. This API enables users to analyze long-term price trends, patterns, and market movements in great detail."
        ),
        params={
            "symbol*": (str, "GCUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "GCUSD",
                "date": "2022-02-04",
                "open": 2850.4,
                "high": 2877.1,
                "low": 2837.4,
                "close": 2873.7,
                "volume": 137844,
                "change": 23.3,
                "changePercent": 0.81743,
                "vwap": 2859.65,
            },
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def commodities_full_chart(self, params: dict) -> dict:
        """
        About Full Chart API
        The FMP Comprehensive Commodities Price API provides detailed historical data for various commodities, including opening, high, low, and closing prices, as well as trading volume and price changes. This API is designed for investors, analysts, and traders who need in-depth market insights to evaluate the performance of commodities over time and make data-driven decisions.

        - Detailed Historical Data: Access historical end-of-day data, including opening, closing, high, and low prices, trading volume, and price changes.
        - Trend Analysis: Analyze long-term price trends and market patterns to better understand the volatility and movement of commodities.
        - Comprehensive View: Evaluate not only price movements but also volume and volatility to get a full picture of market conditions.
        This API is a powerful tool for professionals looking to assess long-term trends and patterns in commodity markets, helping to predict future price movements or develop investment strategies based on historical data.
        """
        return params

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="Interval",
        endpoint="historical-chart/1min",
        name="1-Minute Interval Commodities Chart API",
        description=(
            "Track real-time, short-term price movements for commodities with the FMP 1-Minute Interval Commodities Chart API. This API provides detailed 1-minute interval data, enabling precise monitoring of intraday market changes."
        ),
        params={
            "symbol*": (str, "GCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:19:00",
                "open": 2872,
                "low": 2872,
                "high": 2872.1,
                "close": 2872.1,
                "volume": 4,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def commodities_1min_chart(self, params: dict) -> dict:
        """
        About 1-Minute Interval Commodities Chart API
        The FMP 1-Minute Interval Commodities Chart API delivers minute-by-minute price data for commodities, including open, high, low, and close prices, as well as trading volume. This API is ideal for day traders, analysts, and market participants who require highly granular data to monitor real-time price fluctuations and respond to market trends with speed and accuracy.

        - Real-Time Intraday Data: Access up-to-the-minute price data for commodities, making it easier to track short-term price movements.
        - Detailed Price Information: View open, high, low, and close prices, along with trading volume, for precise analysis of market trends.
        - Fast Decision-Making: The 1-minute interval data supports fast decision-making for intraday trading, allowing users to act on market opportunities as they arise.
        This API is a valuable resource for active traders and investors who need to stay on top of real-time price changes in the fast-moving commodities market.
        """
        return params

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="Interval",
        endpoint="historical-chart/5min",
        name="5-Minute Interval Commodities Chart API",
        description=(
            "Monitor short-term price movements with the FMP 5-Minute Interval Commodities Chart API. This API provides detailed 5-minute interval data, enabling users to track near-term price trends for more strategic trading and investment decisions."
        ),
        params={
            "symbol*": (str, "GCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:15:00",
                "open": 2871.8,
                "low": 2871.7,
                "high": 2872.3,
                "close": 2871.8,
                "volume": 93,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def commodities_5min_chart(self, params: dict) -> dict:
        """
        About 5-Minute Interval Commodities Chart API
        The FMP 5-Minute Interval Commodities Chart API delivers price data at 5-minute intervals, offering a balance between granularity and broader trend analysis. It includes open, high, low, and close prices, as well as trading volume for commodities. This API is ideal for traders and investors who want to track short-term market activity but prefer a slightly broader view than 1-minute data can provide.

        - Short-Term Trend Analysis: Access 5-minute interval data to monitor price movements and identify short-term trends in commodity markets.
        - Detailed Pricing Information: Retrieve detailed price data for each 5-minute interval, including open, high, low, and close prices, along with volume.
        - Strategic Trading: Use the 5-minute interval data to spot patterns and price movements, helping traders refine their strategies and make more informed decisions.
        This API is perfect for traders looking to balance real-time trading needs with a slightly longer-term perspective on commodity market movements.
        """
        return params

    @BaseProxy.endpoint(
        category="Commodity",
        sub_category="Interval",
        endpoint="historical-chart/1hour",
        name="1-Hour Interval Commodities Chart API",
        description=(
            "Monitor hourly price movements and trends with the FMP 1-Hour Interval Commodities Chart API. This API provides hourly data, offering a detailed look at price fluctuations throughout the trading day to support mid-term trading strategies and market analysis."
        ),
        params={
            "symbol*": (str, "GCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:00:00",
                "open": 2872.1,
                "low": 2872,
                "high": 2872.4,
                "close": 2872.4,
                "volume": 66,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def commodities_1hour_chart(self, params: dict) -> dict:
        """
        About 1-Hour Interval Commodities Chart API
        The FMP 1-Hour Interval Commodities Chart API provides access to 1-hour interval pricing data for commodities, including open, high, low, and close prices, along with trading volume. This data is ideal for traders and analysts who need to track hourly trends, offering a balance between short-term and daily price analysis. By focusing on hourly intervals, users can capture significant intraday movements while avoiding the noise of minute-level fluctuations.

        - Hourly Trend Monitoring: Track price movements and trends for commodities with hourly updates, providing a clearer picture of market direction throughout the day.
        - Detailed Pricing Information: Retrieve open, high, low, and close prices for each hour, along with trading volume, to understand market activity during specific time frames.
        - Mid-Term Strategy Support: Use the hourly data to spot intraday trends, helping traders make more informed decisions and refine mid-term strategies.
        This API is a valuable tool for traders, investors, and analysts looking to monitor price trends over the course of the trading day, providing actionable insights for strategic trades.
        """
        return params

    ########################################
    ### Crypto Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="Cryptocurrency",
        endpoint="cryptocurrency-list",
        name="Cryptocurrency List API",
        description=(
            "Access a comprehensive list of all cryptocurrencies traded on exchanges worldwide with the FMP Cryptocurrencies Overview API. Get detailed information on each cryptocurrency to inform your investment strategies."
        ),
        params={},
        response=[
            {
                "symbol": "ALIENUSD",
                "name": "Alien Inu USD",
                "exchange": "CCC",
                "icoDate": "2021-11-22",
                "circulatingSupply": 0,
                "totalSupply": None,
            }
        ],
        dt_cutoff=("icoDate", "%Y-%m-%d"),
    )
    def crypto_list(self, params: dict) -> dict:
        """
        About Cryptocurrency List API
        The FMP Cryptocurrencies Overview API provides detailed information on all cryptocurrencies that are actively traded on global exchanges. This API is essential for:

        - Cryptocurrency Identification: Access a list of all traded cryptocurrencies, including their symbols, names, and the fiat currency they are paired with. This data helps investors identify different cryptocurrencies and understand their market presence.
        - Exchange Details: The API also provides information about the exchange where the cryptocurrency is listed, including the exchange name and a short name identifier. This allows investors to track where each cryptocurrency is traded.
        - Informed Decision-Making: Use the detailed data provided by this API to track cryptocurrency performance, monitor market trends, and make informed investment decisions.
        Example

        Market Analysis: A crypto trader might use the Cryptocurrencies Overview API to compile a list of all cryptocurrencies paired with USD across different exchanges. By analyzing this data, the trader can identify which cryptocurrencies are gaining popularity and may present investment opportunities.
        """
        return params

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="End Of Day",
        endpoint="historical-price-eod/light",
        name="Historical Cryptocurrency Price Snapshot API",
        description=(
            "Access historical end-of-day prices for a variety of cryptocurrencies with the Historical Cryptocurrency Price Snapshot API. Track trends in price and trading volume over time to better understand market behavior."
        ),
        params={
            "symbol*": (str, "BTCUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "BTCUSD",
                "date": "2022-02-04",
                "price": 97347.18,
                "volume": 70745931776,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def crypto_price_snapshot(self, params: dict) -> dict:
        """
        About Historical Cryptocurrency Price Snapshot API
        The Historical Cryptocurrency Price Snapshot API provides crucial insights into the performance of cryptocurrencies over time by offering:

        - End-of-Day Prices: Retrieve historical end-of-day prices for cryptocurrencies, allowing you to analyze long-term market trends and patterns.
        - Trading Volume Data: Access volume data to evaluate market activity during specific time frames.
        - Price Trend Analysis: Use this data to review how a cryptocurrency's value has changed, assisting in making informed investment decisions.
        This API is essential for traders, analysts, and investors looking to perform technical analysis or monitor how the market has evolved over time.

        Example Use Case
        An analyst can use the Historical Cryptocurrency Price Snapshot API to backtest trading strategies by reviewing past price movements and identifying patterns that could influence future price action.
        """
        return params

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="End Of Day",
        endpoint="historical-price-eod/full",
        name="Full Historical Cryptocurrency Data API",
        description=(
            "Access comprehensive end-of-day (EOD) price data for cryptocurrencies with the Full Historical Cryptocurrency Data API. Analyze long-term price trends, market movements, and trading volumes to inform strategic decisions."
        ),
        params={
            "symbol*": (str, "BTCUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "BTCUSD",
                "date": "2022-02-04",
                "open": 101460.15,
                "high": 101812.23,
                "low": 97321.18,
                "close": 97347.18,
                "volume": 70745931776,
                "change": -4112.97,
                "changePercent": -4.05378,
                "vwap": 99485.185,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def crypto_full_chart(self, params: dict) -> dict:
        """
        About Full Historical Cryptocurrency Data API
        The Full Historical Cryptocurrency Data API provides extensive historical data, including:

        - End-of-Day (EOD) Prices: Retrieve daily open, high, low, close (OHLC) price data for cryptocurrencies.
        - Comprehensive Market Data: Access trading volumes, price changes, and VWAP (Volume Weighted Average Price) to gain insights into market behavior.
        - Analyze Long-Term Trends: Review historical price data to track long-term trends, volatility, and market cycles, enabling better decision-making for investors and analysts.
        This API is essential for long-term investors, analysts, and institutions seeking to evaluate market movements, identify trends, and support strategic planning.

        Example Use Case
        A long-term cryptocurrency investor could use the Full Historical Cryptocurrency Data API to analyze Bitcoin’s market performance over the past year, identifying key resistance levels and potential buying opportunities based on historical price trends.
        """
        return params

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="Interval",
        endpoint="historical-chart/1min",
        name="1-Minute Cryptocurrency Intraday Data API",
        description=(
            "Get real-time, 1-minute interval price data for cryptocurrencies with the 1-Minute Cryptocurrency Intraday Data API. Monitor short-term price fluctuations and trading volume to stay updated on market movements."
        ),
        params={
            "symbol*": (str, "BTCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:28:00",
                "open": 98137.6,
                "low": 98098,
                "high": 98263.08,
                "close": 98220.44,
                "volume": 815015.7848495352,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def crypto_1min_chart(self, params: dict) -> dict:
        """
        About 1-Minute Cryptocurrency Intraday Data API
        The 1-Minute Cryptocurrency Intraday Data API offers precise, real-time updates on cryptocurrency price movements, including:

        - 1-Minute Price Intervals: Retrieve data on cryptocurrency prices at 1-minute intervals, including open, high, low, close (OHLC) values.
        - Real-Time Volume Information: Access detailed trading volume data for every minute, enabling quick insights into market activity.
        - Track Short-Term Price Movements: Analyze short-term trends in cryptocurrency prices to capitalize on market opportunities or identify trends early.
        This API is vital for day traders, analysts, and algorithmic traders who need fast, actionable data to track the fast-moving cryptocurrency markets.

        Example Use Case
        A day trader can use the 1-Minute Cryptocurrency Intraday Data API to monitor real-time price movements and volume spikes, making quick decisions based on emerging market trends or breakouts.
        """
        return params

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="Interval",
        endpoint="historical-chart/5min",
        name="5-Minute Interval Cryptocurrency Data API",
        description=(
            "Analyze short-term price trends with the 5-Minute Interval Cryptocurrency Data API. Access real-time, intraday price data for cryptocurrencies to monitor rapid market movements and optimize trading strategies."
        ),
        params={
            "symbol*": (str, "BTCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:25:00",
                "open": 97960.14,
                "low": 97896,
                "high": 98263.08,
                "close": 98220.44,
                "volume": 1699027.774190811,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def crypto_5min_chart(self, params: dict) -> dict:
        """
        About 5-Minute Interval Cryptocurrency Data API

        The 5-Minute Interval Cryptocurrency Data API provides detailed intraday data for cryptocurrencies, including:

        - Short-Term Price Movements: Track prices in 5-minute intervals, offering granular insights into cryptocurrency performance throughout the trading day.
        - Real-Time Market Analysis: Access real-time updates on open, high, low, and close (OHLC) prices, as well as trading volumes, to capture intraday market shifts.
        - Support for Technical Analysis: Use 5-minute interval data to perform advanced technical analysis, such as identifying support and resistance levels, spotting short-term trends, or implementing day trading strategies.
        This API is essential for active traders, analysts, and investors who need to stay informed of fast-moving price changes and capitalize on short-term market fluctuations.

        Example Use Case
        A day trader uses the 5-Minute Interval Cryptocurrency Data API to track Bitcoin's price movements throughout the day. By analyzing the short-term price trends, the trader identifies optimal entry and exit points for their trades.
        """
        return params

    @BaseProxy.endpoint(
        category="Crypto",
        sub_category="Interval",
        endpoint="historical-chart/1hour",
        name="1-Hour Interval Cryptocurrency Data API",
        description=(
            "Access detailed 1-hour intraday price data for cryptocurrencies with the 1-Hour Interval Cryptocurrency Data API. Track hourly price movements to gain insights into market trends and make informed trading decisions throughout the day."
        ),
        params={
            "symbol*": (str, "BTCUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:00:00",
                "open": 97795.06,
                "low": 97761,
                "high": 97919.26,
                "close": 97898.8,
                "volume": 1829413.547367432,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def crypto_1hour_chart(self, params: dict) -> dict:
        """
        About 1-Hour Interval Cryptocurrency Data API
        The 1-Hour Interval Cryptocurrency Data API provides key hourly updates on cryptocurrency prices, offering users a granular view of market fluctuations:

        - Hourly Price Updates: Receive cryptocurrency price data, including open, high, low, and close (OHLC) prices, as well as trading volumes, updated every hour.
        - Comprehensive Market Monitoring: Use hourly data to monitor market trends, track price momentum, and identify potential trading opportunities.
        - Effective for Trend Analysis: Leverage hourly intervals to observe intraday price patterns, helping you make better decisions for day trading, swing trading, or long-term analysis.
        This API is ideal for traders and investors who want a closer look at how prices evolve over the course of a trading day, enabling them to act swiftly in fast-paced markets.

        Example Use Case
        A swing trader uses the 1-Hour Interval Cryptocurrency Data API to monitor the price of Ethereum. By analyzing hourly trends, the trader can spot potential breakouts or pullbacks and adjust their positions accordingly.
        """
        return params

    ########################################
    ### Forex Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="Forex",
        endpoint="forex-list",
        name="Forex Currency Pairs API",
        description=(
            "Access a comprehensive list of all currency pairs traded on the forex market with the FMP Forex Currency Pairs API. Analyze and track the performance of currency pairs to make informed investment decisions."
        ),
        params={},
        response=[
            {
                "symbol": "ARSMXN",
                "fromCurrency": "ARS",
                "toCurrency": "MXN",
                "fromName": "Argentine Peso",
                "toName": "Mexican Peso",
            },
        ],
    )
    def forex_list(self, params: dict) -> dict:
        """
        About Forex Currency Pairs API
        The FMP Forex Currency Pairs API provides detailed information on all currency pairs traded on the global forex market. This API is essential for:

        - Currency Pair Identification: Easily identify the various currency pairs available for trading in the forex market. A currency pair consists of a base currency and a counter currency, with the value of the pair representing how much of the counter currency is needed to purchase one unit of the base currency.
        - Performance Tracking: Use the API to track the performance of different currency pairs over time. This data is crucial for investors and traders looking to monitor market trends and exchange rate movements.
        - Informed Decision-Making: Leverage the comprehensive data provided by the Forex Currency Pairs API to make well-informed decisions when trading currencies. By understanding the dynamics of currency pairs, you can develop strategies that align with market conditions.
        This API is a valuable tool for forex traders, investors, and analysts who need to stay updated on the latest currency pairs and market trends.

        Example
        Forex Trading Strategy: A forex trader might use the Forex Currency Pairs API to identify high-volume currency pairs such as EUR/USD or GBP/JPY. By tracking the performance of these pairs over time, the trader can develop strategies to capitalize on market movements.
        """
        return params

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="End Of Day",
        endpoint="historical-price-eod/light",
        name="Historical Forex Light Chart API",
        description=(
            "Access historical end-of-day forex prices with the Historical Forex Light Chart API. Track long-term price trends across different currency pairs to enhance your trading and analysis strategies."
        ),
        params={
            "symbol*": (str, "EURUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "EURUSD",
                "date": "2022-02-04",
                "price": 1.03791,
                "volume": 297683,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def forex_light_chart(self, params: dict) -> dict:
        """
        About Historical Forex Light Chart API
        The Historical Forex Light Chart API provides end-of-day forex prices for a wide range of currency pairs. This data is invaluable for traders and analysts looking to:

        - Analyze Long-Term Trends: Review historical price data to identify patterns and trends that could influence future market movements.
        - Backtest Trading Strategies: Use past data to validate trading strategies by simulating market conditions over extended timeframes.
        - Compare Forex Pair Performance: Analyze the performance of different forex pairs over time, helping you make more informed trading decisions.
        This API is essential for forex traders, analysts, and investors who need access to accurate historical data for market analysis and strategy development.

        Example Use Case
        A forex trader uses the Historical Forex Light Chart API to review end-of-day prices for the EUR/USD currency pair over the past five years. By analyzing this data, the trader identifies key support and resistance levels, helping refine their trading strategy.
        """
        return params

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="End Of Day",
        endpoint="historical-price-eod/full",
        name="Full Historical Forex Chart API",
        description=(
            "Access comprehensive historical end-of-day forex price data with the Full Historical Forex Chart API. Gain detailed insights into currency pair movements, including open, high, low, close (OHLC) prices, volume, and percentage changes."
        ),
        params={
            "symbol*": (str, "EURUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2023-02-04"),
        },
        response=[
            {
                "symbol": "EURUSD",
                "date": "2022-02-04",
                "open": 1.03432,
                "high": 1.03873,
                "low": 1.02713,
                "close": 1.03791,
                "volume": 297683,
                "change": 0.00359,
                "changePercent": 0.34709,
                "vwap": 1.03452,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def forex_full_chart(self, params: dict) -> dict:
        """
        About Full Historical Forex Chart API
        The Full Historical Forex Chart API provides extensive historical price data for a wide range of currency pairs, offering traders and analysts a deeper understanding of market trends. This data includes open, high, low, and close prices, as well as volume, VWAP (Volume Weighted Average Price), and percentage changes. This API is ideal for:

        - Detailed Trend Analysis: Review comprehensive historical price data to analyze long-term trends and patterns in forex markets.
        - Advanced Technical Analysis: Use OHLC data to apply technical indicators and identify potential trading signals.
        - Strategy Backtesting: Access detailed historical data to validate and optimize trading strategies using real market conditions from past periods.
        This API is an essential resource for traders, analysts, and portfolio managers seeking to understand forex market movements and refine their strategies with comprehensive data.

        Example Use Case
        A portfolio manager uses the Full Historical Forex Chart API to analyze the EUR/USD pair's daily open, high, low, and close prices over the last decade. By reviewing these trends, the manager develops a more informed strategy for managing currency exposure.
        """
        return params

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="Interval",
        endpoint="historical-chart/1min",
        name="1-Minute Forex Interval Chart API",
        description=(
            "Access real-time 1-minute intraday forex data with the 1-Minute Forex Interval Chart API. Track short-term price movements for precise, up-to-the-minute insights on currency pair fluctuations."
        ),
        params={
            "symbol*": (str, "EURUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:29:00",
                "open": 1.03751,
                "low": 1.03737,
                "high": 1.0376,
                "close": 1.0376,
                "volume": 30,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def forex_1min_chart(self, params: dict) -> dict:
        """
        About 1-Minute Forex Interval Chart API
        The 1-Minute Forex Interval Chart API provides high-frequency intraday data, offering a detailed view of currency pair price changes every minute. With real-time open, high, low, close (OHLC) prices and volume data, this API is ideal for:

        - Scalping and Day Trading: Traders focused on quick entry and exit points can leverage minute-by-minute data for highly dynamic market conditions.
        - High-Frequency Monitoring: Closely monitor short-term forex price movements to seize opportunities or manage risk during volatile market sessions.
        - Short-Term Strategy Execution: Apply rapid trading strategies and technical analysis to capture fleeting trends and minimize risk.
        By using this API, traders can make timely and informed decisions in fast-moving forex markets, making it essential for high-frequency traders and those employing short-term strategies.

        Example Use Case
        A day trader uses the 1-Minute Forex Interval Chart API to track price movements in the EUR/USD currency pair. By monitoring each minute’s open, high, low, and close prices, the trader executes a scalping strategy and optimizes profit opportunities within a single trading session.
        """
        return params

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="Interval",
        endpoint="historical-chart/5min",
        name="5-Minute Forex Interval Chart API",
        description=(
            "Track short-term forex trends with the 5-Minute Forex Interval Chart API. Access detailed 5-minute intraday data to monitor currency pair price movements and market conditions in near real-time."
        ),
        params={
            "symbol*": (str, "EURUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:25:00",
                "open": 1.03711,
                "low": 1.03709,
                "high": 1.0376,
                "close": 1.0376,
                "volume": 113,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def forex_5min_chart(self, params: dict) -> dict:
        """
        About 5-Minute Forex Interval Chart API
        The 5-Minute Forex Interval Chart API offers critical price data at 5-minute intervals, making it ideal for traders and analysts focused on short-term trends. With open, high, low, close (OHLC) prices and volume data for each 5-minute period, this API supports:

        - Intraday Trading Strategies: Perfect for traders looking to capture price trends and make informed decisions within short timeframes.
        - Monitoring Currency Pair Volatility: Follow price movements closely during key market sessions to capitalize on fluctuations in exchange rates.
        - Near-Term Trend Analysis: Use this API for technical analysis and to spot patterns or breakouts that occur over 5-minute periods.
        This API is a valuable tool for forex traders aiming to understand and react to market conditions quickly, as well as for analysts seeking to track short-term currency pair movements.

        Example Use Case
        A forex trader monitoring the EUR/USD pair uses the 5-Minute Forex Interval Chart API to analyze price fluctuations during volatile periods. By tracking 5-minute intervals, the trader makes informed decisions on when to enter or exit trades.
        """
        return params

    @BaseProxy.endpoint(
        category="Forex",
        sub_category="Interval",
        endpoint="historical-chart/1hour",
        name="1-Hour Forex Interval Chart API",
        description=(
            "Track forex price movements over the trading day with the 1-Hour Forex Interval Chart API. This tool provides hourly intraday data for currency pairs, giving a detailed view of trends and market shifts."
        ),
        params={
            "symbol*": (str, "EURUSD"),
            "from": (str, "2022-01-01"),
            "to": (str, "2022-03-01"),
        },
        response=[
            {
                "date": "2022-02-04 19:00:00",
                "open": 1.03716,
                "low": 1.03715,
                "high": 1.03743,
                "close": 1.03737,
                "volume": 45,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def forex_1hour_chart(self, params: dict) -> dict:
        """
        About 1-Hour Forex Interval Chart API
        The 1-Hour Forex Interval Chart API delivers comprehensive OHLC (open, high, low, close) price and volume data for each 1-hour period. It’s an essential tool for forex traders and analysts who need to:

        - Monitor Intraday Market Activity: Follow price changes in 1-hour increments throughout the trading day, making it easier to spot trends or reversals.
        - Analyze Long-Term Intraday Patterns: Use 1-hour data to gain insights into the broader movements of currency pairs over the course of the trading day.
        - Support Swing Trading Strategies: With hourly updates, this API is perfect for traders who operate in mid-term strategies, reacting to larger market trends.
        Whether you're actively trading or conducting market analysis, the 1-Hour Forex Interval Chart API helps provide the necessary data to make informed decisions based on evolving market conditions.

        Example Use Case
        A forex analyst looking to optimize their swing trading strategy uses the 1-Hour Forex Interval Chart API to track price movements of the USD/JPY pair. By monitoring hourly changes, the analyst identifies price consolidation points and adjusts their trades accordingly.
        """
        return params

    ########################################
    ### News Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="News",
        sub_category="Articles",
        endpoint="news/general-latest",
        name="General News API",
        description=(
            "Access the latest general news articles from a variety of sources with the FMP General News API. Obtain headlines, snippets, and publication URLs for comprehensive news coverage."
        ),
        params={
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": None,
                "publishedDate": "2022-02-03 23:51:37",
                "publisher": "CNBC",
                "title": "Asia tech stocks rise after Trump pauses tariffs on China and Mexico",
                "image": "https://images.financialmodelingprep.com/news/asia-tech-stocks-rise-20220203.jpg",
                "site": "cnbc.com",
                "text": "Gains in Asian tech companies were broad-based, with stocks in Japan, South Korea and Hong Kong advancing...",
                "url": "https://www.cnbc.com/2022/02/04/asia-tech-stocks-rise...",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def general_news(self, params: dict) -> dict:
        """
        About General News API
        The FMP General News API provides access to the latest general news articles from a wide range of sources. This endpoint includes:

        - Headlines: Stay informed with the latest headlines on current events.
        - Snippets: Get brief summaries of the articles to quickly understand the key points.
        - Publication URLs: Access full articles through provided URLs for detailed information.
        This API is updated daily to ensure you have the most current news. Simply provide the date range you are interested in, and the endpoint will return a list of all general news articles published during that period.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Articles",
        endpoint="news/press-releases-latest",
        name="Press Releases API",
        description=(
            "Access official company press releases with the FMP Press Releases API. Get real-time updates on corporate announcements, earnings reports, mergers, and more."
        ),
        params={
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "LNW",
                "publishedDate": "2022-02-03 23:32:00",
                "publisher": "PRNewsWire",
                "title": "Rosen Law Firm Encourages Light & Wonder, Inc. Investors to Inquire About Securities Class Action Investigation - LNW",
                "image": "https://images.financialmodelingprep.com/news/rosen-law-firm-encourages-20220203.jpg",
                "site": "prnewswire.com",
                "text": "NEW YORK , Feb. 3, 2022 /PRNewswire/ -- Rosen Law Firm continues to investigate potential securities claims on behalf of Light & Wonder shareholders...",
                "url": "https://www.prnewswire.com/news-releases/...",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def press_releases(self, params: dict) -> dict:
        """
        About Press Releases API
        The Press Releases API provides real-time access to official company announcements, allowing investors, analysts, and business professionals to stay informed on the latest developments. This API is crucial for:

        - Company Announcements: Stay informed about earnings reports, product launches, mergers, and more directly from companies.
        - Strategic Updates: Track leadership changes, business restructuring, and other significant corporate strategies that may affect a company's market standing.
        - Market Impact Analysis: Analyze how company press releases influence stock prices, company valuations, and market sentiment.
        This API ensures that you have access to the most current press releases, helping you make informed decisions based on the latest corporate disclosures.

        Example Use Case
        A financial analyst uses the Press Releases API to monitor corporate announcements from publicly traded companies, providing critical insights for investment decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Articles",
        endpoint="news/stock-latest",
        name="Stock News API",
        description=(
            "Stay informed with the latest stock market news using the FMP Stock News Feed API. Access headlines, snippets, publication URLs, and ticker symbols for the most recent articles from a variety of sources."
        ),
        params={
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 500),
        },
        response=[
            {
                "symbol": "INSG",
                "publishedDate": "2022-02-03 23:53:40",
                "publisher": "Seeking Alpha",
                "title": "Q4 Earnings Release Looms For Inseego, But Don't Expect Miracles",
                "image": "https://images.financialmodelingprep.com/news/q4-earnings-release-looms-20220203.jpg",
                "site": "seekingalpha.com",
                "text": "Inseego's Q3 beat was largely due to a one-time debt restructuring gain...",
                "url": "https://seekingalpha.com/article/4754485-inseego-stock-q4-earnings-preview...",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def stock_news(self, params: dict) -> dict:
        """
        About Stock News API
        The Stock News API offers up-to-date information on stock market events, keeping traders, investors, and financial professionals informed about:

        - Breaking Market News: Access the latest headlines that may impact stock prices and market movements.
        - Company-Specific News: Stay updated on news related to individual stocks, including earnings reports, product announcements, and mergers.
        - Market Trends and Analysis: Follow broader market trends and sentiment to make better investment decisions.
        This API is designed to provide timely news that helps professionals track stock market developments and make informed decisions.

        Example Use Case
        A portfolio manager uses the Stock News API to track real-time updates on the stock markets, ensuring they are aware of any news that may affect the performance of the equities in their portfolio.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Articles",
        endpoint="news/crypto-latest",
        name="Crypto News API",
        description=(
            "Stay informed with the latest cryptocurrency news using the FMP Crypto News API. Access a curated list of articles from various sources, including headlines, snippets, and publication URLs."
        ),
        params={
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "BTCUSD",
                "publishedDate": "2022-02-03 23:32:19",
                "publisher": "Coingape",
                "title": "Crypto Prices Today Feb 4: BTC & Altcoins Recover Amid Pause On Trump's Tariffs",
                "image": "https://images.financialmodelingprep.com/news/crypto-prices-today-feb-4-btc-altcoins-20220203.webp",
                "site": "coingape.com",
                "text": "Crypto prices today have shown signs of recovery as President Trump's tariffs were paused...",
                "url": "https://coingape.com/crypto-prices-today-feb-4-btc-altcoins-recover-amid-pause-on-trumps-tariffs/",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def crypto_news(self, params: dict) -> dict:
        """
        About Crypto News API
        The Crypto News API provides up-to-date news on cryptocurrencies, including key market events and trends. This API is critical for:

        - Real-Time Updates: Receive the latest news on major cryptocurrencies like Bitcoin, Ethereum, and more.
        - Market Sentiment Analysis: Follow news and reports that could influence crypto market sentiment and price movements.
        - Cryptocurrency Trends: Stay informed about industry developments, new technologies, and regulatory updates.
        This API is a must-have for anyone involved in the fast-moving world of cryptocurrency investing and trading.

        Example Use Case
        A crypto trader uses the Crypto News API to track daily news on Bitcoin and Ethereum, enabling them to stay ahead of market trends.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Articles",
        endpoint="news/forex-latest",
        name="Forex News API",
        description=(
            "Stay updated with the latest forex news articles from various sources using the FMP Forex News API. Access headlines, snippets, and publication URLs for comprehensive market insights."
        ),
        params={
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "XAUUSD",
                "publishedDate": "2022-02-03 23:55:44",
                "publisher": "FX Street",
                "title": "United Arab Emirates Gold price today: Gold steadies, according to FXStreet data",
                "image": "https://images.financialmodelingprep.com/news/united-arab-emirates-gold-price-today-20220203.jpg",
                "site": "fxstreet.com",
                "text": "Gold prices remained broadly unchanged in the UAE, according to FXStreet data.",
                "url": "https://www.fxstreet.com/news/united-arab-emirates-gold-price-today-202202040455",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def forex_news(self, params: dict) -> dict:
        """
        About Forex News API
        The Forex News API provides up-to-date reports on currency markets, ensuring you stay informed about:

        - Currency Market Movements: Get real-time updates on the forex market, including major events and macro-economic trends that influence currency pairs.
        - Currency Pair Analysis: Stay informed on specific currency pair movements, such as EUR/USD, GBP/USD, or JPY/CHF, to better understand market conditions.
        - Market Sentiment Updates: Follow forex-related news to gauge investor sentiment and market dynamics in the foreign exchange sector.
        This API is essential for traders, analysts, and financial professionals who need to stay on top of the ever-changing forex markets.

        Example Use Case
        A forex trader uses the Forex News API to track the latest news on currency pairs, helping them make quick and informed trading decisions.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Symbol",
        endpoint="news/press-releases",
        name="Search Press Releases API",
        description=(
            "Search for company press releases with the FMP Search Press Releases API. Find specific corporate announcements and updates by entering a stock symbol or company name."
        ),
        params={
            "symbols*": (str, "AAPL,GOOGL,AMZN"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "AAPL",
                "publishedDate": "2022-01-30 16:30:00",
                "publisher": "Business Wire",
                "title": "Apple reports first quarter results",
                "image": "https://images.financialmodelingprep.com/news/apple-reports-first-quarter-results-20220130.jpg",
                "site": "businesswire.com",
                "text": "CUPERTINO, Calif.--(BUSINESS WIRE)--Apple® today announced financial results for its fiscal 2022 first quarter...",
                "url": "https://www.businesswire.com/news-releases/Apple-reports-first-quarter-results/",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def search_press_releases(self, params: dict) -> dict:
        """
        About Search Press Releases API
        The Search Press Releases API allows users to find specific press releases based on a company name or stock symbol, offering quick access to relevant announcements. This API is essential for:

        - Targeted Searches: Narrow down your search to find exact press releases from a particular company.
        - Symbol-Based Retrieval: Use stock symbols to pinpoint corporate disclosures, making it ideal for investors and analysts looking for precise data.
        - Historical and Real-Time Access: Retrieve both current and past press releases, helping with long-term trend analysis.
        This API is designed for professionals who need quick, reliable access to specific press releases, saving time and providing accurate data.

        Example Use Case
        An investor uses the Search Press Releases API to find the most recent earnings report of a specific company before making an investment decision.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Symbol",
        endpoint="news/stock",
        name="Search Stock News API",
        description=(
            "Search for stock-related news using the FMP Search Stock News API. Find specific stock news by entering a ticker symbol or company name to track the latest developments."
        ),
        params={
            "symbols*": (str, "AAPL"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 500),
        },
        response=[
            {
                "symbol": "AAPL",
                "publishedDate": "2022-02-03 21:05:14",
                "publisher": "Zacks Investment Research",
                "title": "Apple & China Tariffs: A Closer Look",
                "image": "https://images.financialmodelingprep.com/news/apple-china-tariffs-20220203.jpg",
                "site": "zacks.com",
                "text": "Tariffs have been the talk of the town over recent weeks...",
                "url": "https://www.zacks.com/stock/news/2408814/apple-china-tariffs-a-closer-look",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def search_stock_news(self, params: dict) -> dict:
        """
        About Search Stock News API
        The Search Stock News API helps users find stock-related news by entering a specific company name or stock symbol. This tool is ideal for:

        - Targeted News Searches: Narrow down your search to find news about specific companies or stocks.
        - Symbol-Based Lookup: Quickly retrieve news by entering the relevant ticker symbol for a stock.
        - Comprehensive News Retrieval: Access both current and historical news reports to gain a full picture of stock movements over time.
        This API is tailored for investors and analysts who require fast, reliable access to news affecting specific stocks.

        Example Use Case
        A trader uses the Search Stock News API to look up recent news articles about a stock they are considering buying, helping them make an informed decision.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Symbol",
        endpoint="news/crypto",
        name="Search Crypto News API",
        description=(
            "Search for cryptocurrency news using the FMP Search Crypto News API. Retrieve news related to specific coins or tokens by entering their name or symbol."
        ),
        params={
            "symbols*": (str, "BTCUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "BTCUSD",
                "publishedDate": "2022-02-03 23:32:19",
                "publisher": "Coingape",
                "title": "Crypto Prices Today Feb 4: BTC & Altcoins Recover Amid Pause On Trump's Tariffs",
                "image": "https://images.financialmodelingprep.com/news/crypto-prices-today-feb-4-20220203.webp",
                "site": "coingape.com",
                "text": "Crypto prices today have shown signs of recovery...",
                "url": "https://coingape.com/crypto-prices-today-feb-4...",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def search_crypto_news(self, params: dict) -> dict:
        """
        About Search Crypto News API
        The Search Crypto News API allows users to look up cryptocurrency news by entering a coin name or symbol. This API is helpful for:

        - Targeted Searches: Quickly find news on specific cryptocurrencies by entering their name or ticker symbol.
        - Real-Time & Historical News: Retrieve both current and past news on digital assets to track market trends and price drivers.
        - Symbol-Based Lookups: Find news related to your preferred coins, such as Bitcoin (BTC) or Ethereum (ETH).
        This API is ideal for cryptocurrency investors who need fast access to news that could affect the value of their digital assets.

        Example Use Case
        A crypto investor uses the Search Crypto News API to search for news on Ethereum to understand the recent market movements before making a trade.
        """
        return params

    @BaseProxy.endpoint(
        category="News",
        sub_category="Symbol",
        endpoint="news/forex",
        name="Search Forex News API",
        description=(
            "Search for foreign exchange news using the FMP Search Forex News API. Find targeted news on specific currency pairs by entering their symbols for focused updates."
        ),
        params={
            "symbols*": (str, "EURUSD"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
            "page": (int, 0),
            "limit": (int, 20),
        },
        response=[
            {
                "symbol": "EURUSD",
                "publishedDate": "2022-02-03 18:43:01",
                "publisher": "FX Street",
                "title": "EUR/USD trims losses but still sheds weight",
                "image": "https://images.financialmodelingprep.com/news/eurusd-trims-losses-20220203.jpg",
                "site": "fxstreet.com",
                "text": "EUR/USD dropped sharply following fresh tariff threats...",
                "url": "https://www.fxstreet.com/news/eur-usd-trims-losses...",
            },
        ],
        dt_cutoff=("publishedDate", "%Y-%m-%d %H:%M:%S"),
    )
    def search_forex_news(self, params: dict) -> dict:
        """
        About Search Forex News API
        The Search Forex News API allows users to look up forex news by entering a currency pair, such as EUR/USD or GBP/USD. This API is perfect for:

        - Targeted News Search: Easily find news about specific currency pairs to track the latest developments in the forex market.
        - Historical News Access: Look up both current and historical forex news to analyze long-term trends and market movements.
        - Symbol-Based Retrieval: Enter specific currency pair symbols to retrieve relevant news for informed decision-making.
        This API is ideal for forex traders who need quick access to news related to specific currency pairs.

        Example Use Case
        A currency trader uses the Search Forex News API to search for the latest news on EUR/USD, helping them understand recent price fluctuations before entering a trade.
        """
        return params

    ########################################
    ### Technical Indicators Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/sma",
        name="Simple Moving Average API",
        description="Access the Simple Moving Average (SMA) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "sma": 231.215,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def sma(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/ema",
        name="Exponential Moving Average API",
        description="Access the Exponential Moving Average (EMA) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "ema": 232.8406611792779,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def ema(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/wma",
        name="Weighted Moving Average API",
        description="Access the Weighted Moving Average (WMA) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "wma": 233.04745454545454,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def wma(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/dema",
        name="Double Exponential Moving Average API",
        description="Access the Double Exponential Moving Average (DEMA) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2022-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "dema": 232.10592058582725,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def dema(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/tema",
        name="Triple Exponential Moving Average API",
        description="Access the Triple Exponential Moving Average (TEMA) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2023-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "tema": 233.66383715917516,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def tema(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/rsi",
        name="Relative Strength Index API",
        description="Access the Relative Strength Index (RSI) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2023-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "rsi": 47.64507340768903,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def rsi(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/standarddeviation",
        name="Standard Deviation API",
        description="Access the standard deviation for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2023-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "standardDeviation": 6.139182763202282,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def standard_deviation(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/williams",
        name="Williams API",
        description="Access the Williams %R indicator for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2023-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "williams": -52.51824817518242,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def williams(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Technical Indicators",
        endpoint="technical-indicators/adx",
        name="Average Directional Index API",
        description="Access the Average Directional Index (ADX) for a given symbol over a specified period and timeframe.",
        params={
            "symbol*": (str, "AAPL"),
            "periodLength*": (int, 10),
            "timeframe*": (str, "1day"),
            "from": (str, "2022-11-04"),
            "to": (str, "2022-02-04"),
        },
        response=[
            {
                "date": "2023-02-04 00:00:00",
                "open": 227.2,
                "high": 233.13,
                "low": 226.65,
                "close": 232.8,
                "volume": 44489128,
                "adx": 26.414065772772613,
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d %H:%M:%S"),
    )
    def adx(self, params: dict) -> dict:
        """ """
        return params

    ########################################
    ### SEC Filings Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="SEC Filings",
        sub_category="Company Info",
        endpoint="sec-profile",
        name="SEC Company Full Profile API",
        description="Retrieve detailed company profiles, including business descriptions, executive details, contact information, and financial data with the FMP SEC Company Full Profile API.",
        params={"symbol*": (str, "AAPL"), "limit": (int, 100)},
        response=[
            {
                "symbol": "AAPL",
                "cik": "0000320193",
                "registrantName": "Apple Inc.",
                "sicCode": "3571",
                "sicDescription": "Electronic Computers",
                "sicGroup": "Consumer Electronics",
                "isin": "US0378331005",
                "businessAddress": "ONE APPLE PARK WAY,CUPERTINO CA 95014,(408) 996-1010",
                "mailingAddress": "ONE APPLE PARK WAY,CUPERTINO CA 95014",
                "phoneNumber": "(408) 996-1010",
                "postalCode": "95014",
                "city": "Cupertino",
                "state": "CA",
                "country": "US",
                "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. ...",
                "ceo": "Mr. Timothy D. Cook",
                "website": "https://www.apple.com",
                "exchange": "NASDAQ",
                "stateLocation": "CA",
                "stateOfIncorporation": "CA",
                "fiscalYearEnd": "09-28",
                "ipoDate": "1980-12-12",
                "employees": "164000",
                "secFilingsUrl": "https://www.sec.gov/cgi-bin/browse-edgar?CIK=0000320193",
                "taxIdentificationNumber": "94-2404110",
                "fiftyTwoWeekRange": "164.08 - 260.1",
                "isActive": True,
                "assetType": "stock",
                "openFigiComposite": "BBG000B9XRY4",
                "priceCurrency": "USD",
                "marketSector": "Technology",
                "securityType": None,
                "isEtf": False,
                "isAdr": False,
                "isFund": False,
            }
        ],
        # dt_cutoff=('date', '%Y-%m-%d %H:%M:%S')
    )
    def sec_profile(self, params: dict) -> dict:
        """
        About SEC Company Full Profile API
        The FMP SEC Company Full Profile API offers comprehensive data on companies registered with the SEC. This API is ideal for:

        - Detailed Company Profiles: Access in-depth information on a company's operations, SIC code, CEO, fiscal year, and employee count.
        - Executive and Contact Information: Retrieve key executive details and contact information, including business and mailing addresses, phone numbers, and website links.
        - Company Description and Operations: Get a detailed company description, including its products, services, markets, and business sectors, allowing for a full understanding of its operations.
        - Financial and Regulatory Data: This API provides essential financial data like fiscal year end, IPO date, and links to SEC filings.
        This API is crucial for investors, analysts, and researchers who need detailed corporate profiles for financial analysis, competitive research, and investment decision-making.
        """
        return params

    @BaseProxy.endpoint(
        category="SEC Filings",
        sub_category="Industry Classification",
        endpoint="standard-industrial-classification-list",
        name="Industry Classification List API",
        description="Retrieve a comprehensive list of industry classifications, including Standard Industrial Classification (SIC) codes and industry titles with the FMP Industry Classification List API.",
        params={"industryTitle": (str, "SERVICES"), "sicCode": (str, "7371")},
        response=[
            {
                "office": "Office of Life Sciences",
                "sicCode": "100",
                "industryTitle": "AGRICULTURAL PRODUCTION-CROPS",
            }
        ],
    )
    def standard_industrial_classification_list(self, params: dict) -> dict:
        """
        About Industry Classification List API
        The FMP Industry Classification List API provides a complete directory of SIC codes and corresponding industry titles. This API is essential for:

        - Industry Research: Access an organized list of industries with SIC codes, allowing users to categorize companies based on their industry sector.
        - Company Classification: Retrieve SIC codes for industries ranging from manufacturing to services, helping users classify and analyze companies by their primary business activities.
        - Standardized Data: Ensure consistency when researching or classifying companies, as this API provides standardized SIC codes and official industry titles.
        This API is ideal for analysts, researchers, and businesses looking to categorize companies based on industry standards.
        """
        return params

    @BaseProxy.endpoint(
        category="SEC Filings",
        sub_category="Industry Classification",
        endpoint="industry-classification-search",
        name="Industry Classification Search API",
        description="Search and retrieve industry classification details for companies, including SIC codes, industry titles, and business information, with the FMP Industry Classification Search API.",
        params={
            "symbol": (str, "AAPL"),
            "cik": (str, "320193"),
            "sicCode": (str, "3571"),
        },
        response=[
            {
                "symbol": "AAPL",
                "name": "APPLE INC.",
                "cik": "0000320193",
                "sicCode": "3571",
                "industryTitle": "ELECTRONIC COMPUTERS",
                "businessAddress": "['ONE APPLE PARK WAY', 'CUPERTINO CA 95014']",
                "phoneNumber": "(408) 996-1010",
            }
        ],
    )
    def industry_classification_search(self, params: dict) -> dict:
        """
        About Industry Classification Search API
        The FMP Industry Classification Search API allows users to search for company information based on their Standard Industrial Classification (SIC) codes. This API provides:

        - Company Lookup by Industry: Search for companies by industry classifications, retrieving details such as SIC codes, industry titles, and company contact information.
        - Business Information Access: Get comprehensive company information, including business addresses and phone numbers, making it easier to identify and classify businesses by their industry.
        - SIC Code Matching: Use this API to match companies with their corresponding industry sectors, enhancing your ability to perform industry-specific research and classification.
        This API is valuable for businesses, investors, and researchers who need detailed company information tied to specific industry sectors.
        """
        return params

    ########################################
    ### Earnings Transcript Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Earnings Transcripts",
        endpoint="earning-call-transcript",
        name="Earnings Transcript API",
        description="Access the full transcript of a company's earnings call with the FMP Earnings Transcript API. Stay informed about a company's financial performance, future plans, and overall strategy by analyzing management's communication.",
        params={
            "symbol*": (str, "AAPL"),
            "limit": (int, 100),
            "quarter": (str, "3"),
            "year": (str, "2020"),
        },
        response=[
            {
                "symbol": "AAPL",
                "period": "Q3",
                "year": 2020,
                "date": "2020-07-30",
                "content": "Operator: Good day, everyone. Welcome to the Apple Incorporated Third Quarter Fiscal Year 2020 Earnings Conference Call. Today's call is being recorded. At this time, for opening remarks and introductions, I would like to turn things over to Mr. Tejas Gala, Senior Manager, Corporate Finance and Investor Relations. Please go ahead, sir. ... ",
            }
        ],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def earning_call_transcript(self, params: dict) -> dict:
        """
        About Earnings Transcript API
        The FMP Earnings Transcript API provides complete access to the text transcript of a company’s earnings call. This API is essential for:

        - In-Depth Financial Analysis: Gain valuable insights into a company’s financial performance by reviewing what executives say during earnings calls. The transcript can provide context and details beyond what’s available in standard financial reports.
        - Strategic Planning: Learn about a company’s future plans and strategic direction straight from management. Understanding the company’s priorities and challenges can help investors make informed decisions.
        - Risk Identification: Use the transcript to identify any potential red flags or areas of concern that might not be immediately apparent in the earnings report. This can include management's tone, response to analysts' questions, or any mention of operational or financial difficulties.
        Example Use Case
        Investor Insight: An investor might use the Earnings Transcript API to review the most recent earnings call for a retail company. By analyzing the transcript, the investor can assess the company’s response to market trends, management’s outlook on upcoming quarters, and any potential risks that were discussed.
        """
        return params

    @BaseProxy.endpoint(
        category="Earnings Transcripts",
        endpoint="earning-call-transcript-dates",
        name="Transcripts Dates By Symbol API",
        description="Access earnings call transcript dates for specific companies with the FMP Transcripts Dates By Symbol API. Get a comprehensive overview of earnings call schedules based on fiscal year and quarter.",
        params={
            "symbol*": (str, "AAPL"),
        },
        response=[{"quarter": 1, "fiscalYear": 2022, "date": "2022-01-30"}],
        dt_cutoff=("date", "%Y-%m-%d"),
    )
    def earning_call_transcript_dates(self, params: dict) -> dict:
        """
        About Transcripts Dates By Symbol API
        The FMP Transcripts Dates By Symbol API provides users with precise information about when earnings call transcripts are available for a given company. This API is ideal for investors, analysts, and researchers who want to track earnings discussions and financial insights over time, including:

        - Earnings Call Availability by Quarter: Retrieve transcript dates by quarter and fiscal year to track a company's performance.
        - Timely Access to Transcripts: Get access to transcripts for upcoming or historical earnings calls for in-depth analysis.
        - Comprehensive Coverage: Identify and analyze earnings call transcripts across multiple quarters for better decision-making.
        This API is designed to help users stay informed about earnings call schedules and access key financial insights through transcripts from specific periods.

        Example Use Case
        An investment firm can use the Transcripts Dates By Symbol API to keep track of a company's earnings calls for each quarter and access these transcripts for detailed performance analysis and strategic planning.
        """
        return params

    @BaseProxy.endpoint(
        category="Earnings Transcripts",
        endpoint="earnings-transcript-list",
        name="Available Transcript Symbols API",
        description="Access a complete list of stock symbols with available earnings call transcripts using the FMP Available Earnings Transcript Symbols API. Retrieve information on which companies have earnings transcripts and how many are accessible for detailed financial analysis.",
        params={},
        response=[
            {"symbol": "MCUJF", "companyName": "Medicure Inc.", "noOfTranscripts": "16"}
        ],
    )
    def earnings_transcript_list(self, params: dict) -> dict:
        """
        About Available Transcript Symbols API
        The FMP Available Earnings Transcript Symbols API provides users with a comprehensive list of companies that have earnings call transcripts available. This API is designed for analysts, investors, and researchers who want to track corporate earnings discussions and performance over time, including:

        - Earnings Transcript Availability: Get a list of companies with earnings call transcripts available for review.
        - Number of Available Transcripts: View the total number of transcripts available for each company, allowing users to analyze trends across multiple periods.
        - Quick Access to Relevant Symbols: Easily identify which companies provide insights through earnings calls, facilitating research and performance analysis.
        This API simplifies the process of discovering which companies have earnings transcripts, making it easier to access and analyze financial discussions.

        Example Use Case
        A research analyst can use the Available Earnings Transcript Symbols API to compile a list of companies with multiple earnings transcripts, allowing them to focus on companies with the most available historical data for better trend analysis.
        """
        return params

    ########################################
    ### Senate Endpoints
    ########################################

    @BaseProxy.endpoint(
        category="Senate",
        sub_category="Symbol",
        endpoint="senate-trades",
        name="Senate Trading Activity API",
        description="Monitor the trading activity of US Senators with the FMP Senate Trading Activity API. Access detailed information on trades made by Senators, including trade dates, assets, amounts, and potential conflicts of interest.",
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "AAPL",
                "disclosureDate": "2022-01-08",
                "transactionDate": "2022-12-19",
                "firstName": "Sheldon",
                "lastName": "Whitehouse",
                "office": "Sheldon Whitehouse",
                "district": "RI",
                "owner": "Self",
                "assetDescription": "Apple Inc",
                "assetType": "Stock",
                "type": "Sale (Partial)",
                "amount": "$15,001 - $50,000",
                "capitalGainsOver200USD": "False",
                "comment": "--",
                "link": "https://efdsearch.senate.gov/search/view/ptr/70c80513-d89a-4382-afa6-d80f6c1fcbf1/",
            }
        ],
        dt_cutoff=("disclosureDate", "%Y-%m-%d"),
    )
    def senate_trades(self, params: dict) -> dict:
        """
        About Senate Trading Activity API
        The FMP Senate Trading Activity API provides comprehensive data on the trading activities of US Senators, as required by the STOCK Act of 2012. This API is essential for:

        - Transparency & Accountability: Access a detailed list of trades made by US Senators, including the date, asset, amount traded, and price per share. This transparency helps ensure accountability and provides insights into the financial activities of elected officials.
        - Conflict of Interest Identification: Use the data to identify potential conflicts of interest by analyzing trades made by Senators in companies or sectors where they may have legislative influence. This information is crucial for investors who want to ensure ethical investment practices.
        - Informed Investment Decisions: Investors can track the trading activities of Senators to gain insights into market trends or to flag any trades that might indicate a significant market move. Knowing when and what Senators are trading can provide a unique perspective on market sentiment.
        This API is a powerful tool for investors, analysts, and anyone interested in monitoring the financial activities of US Senators and ensuring transparency in government.

        Example Use Case
        Ethical Investing: An investor focused on ethical investing might use the Senate Trading Activity API to avoid investing in companies where Senators have made trades, especially if those trades could be seen as conflicts of interest. By doing so, the investor aligns their portfolio with ethical standards.
        """
        return params

    @BaseProxy.endpoint(
        category="Senate",
        sub_category="Symbol",
        endpoint="senate-trades-by-name",
        name="Senate Trades By Name API",
        description="Search for Senate trading activity by name using the FMP Senate Trades By Name API. Retrieve detailed trade information by specifying a Senator’s name.",
        params={"name*": (str, "Jerry")},
        response=[
            {
                "symbol": "BRK/B",
                "disclosureDate": "2022-01-18",
                "transactionDate": "2022-12-16",
                "firstName": "Jerry",
                "lastName": "Moran",
                "office": "Jerry Moran",
                "district": "KS",
                "owner": "Self",
                "assetDescription": "Berkshire Hathaway Inc",
                "assetType": "Stock",
                "type": "Purchase",
                "amount": "$1,001 - $15,000",
                "capitalGainsOver200USD": "False",
                "comment": "",
                "link": "https://efdsearch.senate.gov/search/view/ptr/e37322e3-0829-4e3c-9faf-7a4a1a957e09/",
            }
        ],
        dt_cutoff=("disclosureDate", "%Y-%m-%d"),
    )
    def senate_trades_by_name(self, params: dict) -> dict:
        """ """
        return params

    @BaseProxy.endpoint(
        category="Senate",
        sub_category="Symbol",
        endpoint="house-trades",
        name="U.S. House Trades API",
        description="Track the financial trades made by U.S. House members and their families with the FMP U.S. House Trades API. Access real-time information on stock sales, purchases, and other investment activities to gain insight into their financial decisions.",
        params={"symbol*": (str, "AAPL")},
        response=[
            {
                "symbol": "AAPL",
                "disclosureDate": "2022-01-20",
                "transactionDate": "2022-12-31",
                "firstName": "Nancy",
                "lastName": "Pelosi",
                "office": "Nancy Pelosi",
                "district": "CA11",
                "owner": "Spouse",
                "assetDescription": "Apple Inc",
                "assetType": "Stock",
                "type": "Sale",
                "amount": "$10,000,001 - $25,000,000",
                "capitalGainsOver200USD": "False",
                "comment": "",
                "link": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2022/20026590.pdf",
            }
        ],
        dt_cutoff=("disclosureDate", "%Y-%m-%d"),
    )
    def house_trades(self, params: dict) -> dict:
        """
        About U.S. House Trades API
        The FMP U.S. House Trades API provides a comprehensive view of the trading activities of U.S. House members and their spouses. This API offers detailed data on trades, including stock sales and purchases, ownership details, and transaction amounts. Users can:

        - Monitor Trading Activity: Stay informed about the latest stock trades made by U.S. House members and their families.
        - Understand Financial Moves: Gain insights into the financial decisions of government officials through detailed trade data.
        - Transparency and Accountability: Use the data to follow the financial actions of U.S. House members, ensuring greater transparency in government.
        This API is ideal for political analysts, journalists, and the general public interested in understanding the financial moves of U.S. House representatives.
        """
        return params

    @BaseProxy.endpoint(
        category="Senate",
        sub_category="Symbol",
        endpoint="house-trades-by-name",
        name="House Trades By Name API",
        description="Search for U.S. House trading activity by name using the FMP House Trades By Name API. Retrieve trade details by specifying the name of a House member.",
        params={"name*": (str, "James")},
        response=[
            {
                "symbol": "LUV",
                "disclosureDate": "2022-01-13",
                "transactionDate": "2022-12-31",
                "firstName": "James",
                "lastName": "Comer",
                "office": "James Comer",
                "district": "KY01",
                "owner": "",
                "assetDescription": "Southwest Airlines Co",
                "assetType": "Stock",
                "type": "Sale",
                "amount": "$1,001 - $15,000",
                "capitalGainsOver200USD": "False",
                "comment": "",
                "link": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2022/20018054.pdf",
            }
        ],
        dt_cutoff=("disclosureDate", "%Y-%m-%d"),
    )
    def house_trades_by_name(self, params: dict) -> dict:
        """ """
        return params
