"""Search proxy for pkg_delta — a mock API proxy for testing."""
from lllm.proxies.base import BaseProxy


class SearchProxy(BaseProxy):
    """Mock search engine proxy for unit testing."""

    _proxy_path = "search"
    _proxy_name = "Search Proxy"
    _proxy_description = "A mock proxy for searching and indexing documents"

    @BaseProxy.endpoint(
        category="search",
        endpoint="/search",
        description="Search for documents matching a query",
        params={
            "query": (str, "example search query"),
            "limit": (int, 10),
            "offset": (int, 0),
        },
        response=[{"id": "str", "content": "str", "score": "float"}],
        method="GET",
    )
    def search(self, query: str, limit: int = 10, offset: int = 0):
        """Search for documents matching the query string."""
        return [
            {"id": f"doc_{i+offset}", "content": f"Result {i} for '{query}'", "score": 1.0 - i * 0.05}
            for i in range(min(limit, 5))
        ]

    @BaseProxy.endpoint(
        category="index",
        endpoint="/index",
        description="Index a new document into the search engine",
        params={
            "content": (str, "document text to index"),
            "doc_id": (str, "optional_doc_id"),
            "metadata": (dict, {}),
        },
        response=[{"indexed_id": "str", "success": "bool"}],
        method="POST",
    )
    def index(self, content: str, doc_id: str = None, metadata: dict = None):
        """Index a new document into the search engine."""
        _id = doc_id or f"auto_{abs(hash(content)) % 100000}"
        return {"indexed_id": _id, "success": True}

    @BaseProxy.endpoint(
        category="stats",
        endpoint="/stats",
        description="Retrieve index statistics",
        params={},
        response=[{"total_docs": "int", "index_size_mb": "float", "last_updated": "str"}],
        method="GET",
    )
    def get_stats(self):
        """Get statistics about the search index."""
        return {"total_docs": 1000, "index_size_mb": 45.2, "last_updated": "2024-01-01"}

    @BaseProxy.endpoint(
        category="search",
        endpoint="/suggest",
        description="Get query suggestions for autocomplete",
        params={"prefix": (str, "que"), "max_suggestions": (int, 5)},
        response=[{"suggestion": "str", "score": "float"}],
        method="GET",
    )
    def suggest(self, prefix: str, max_suggestions: int = 5):
        """Get autocomplete suggestions for a query prefix."""
        suggestions = ["query", "question", "quick", "qualify", "quota"]
        return [
            {"suggestion": s, "score": 1.0 - i * 0.1}
            for i, s in enumerate(suggestions[:max_suggestions])
            if s.startswith(prefix)
        ]


class AnalyticsProxy(BaseProxy):
    """Mock analytics proxy — tests multiple proxies in one file."""

    _proxy_path = "analytics"
    _proxy_name = "Analytics Proxy"
    _proxy_description = "A mock proxy for analytics events"

    @BaseProxy.endpoint(
        category="events",
        endpoint="/track",
        description="Track an analytics event",
        params={"event_name": (str, "click"), "properties": (dict, {})},
        response=[{"tracked": "bool", "event_id": "str"}],
        method="POST",
    )
    def track(self, event_name: str, properties: dict = None):
        """Track an analytics event with optional properties."""
        return {"tracked": True, "event_id": f"evt_{abs(hash(event_name)) % 9999}"}

    @BaseProxy.endpoint(
        category="events",
        endpoint="/query",
        description="Query analytics events",
        params={"event_name": (str, "click"), "start_date": (str, "2024-01-01"), "end_date": (str, "2024-12-31")},
        response=[{"count": "int", "breakdown": "list"}],
        method="GET",
    )
    def query_events(self, event_name: str, start_date: str = None, end_date: str = None):
        """Query aggregated analytics events."""
        return {"count": 42, "breakdown": [{"date": "2024-01-01", "count": 10}]}
