from lllm.proxies.base import BaseProxy


class SampleDataProxy(BaseProxy):
    _proxy_path = "sample"
    _proxy_name = "Sample Data Proxy"
    _proxy_description = "Example local API surface for account and event data."

    @BaseProxy.endpoint(
        category="accounts",
        endpoint="/accounts",
        description="List known accounts",
        params={"limit": (int, 10)},
        response=[{"id": "str", "name": "str", "tier": "str"}],
        method="GET",
    )
    def list_accounts(self, limit: int = 10):
        accounts = [
            {"id": "acct_001", "name": "Northwind", "tier": "enterprise"},
            {"id": "acct_002", "name": "Contoso", "tier": "startup"},
            {"id": "acct_003", "name": "Fabrikam", "tier": "growth"},
        ]
        return accounts[:limit]

    @BaseProxy.endpoint(
        category="events",
        endpoint="/events",
        description="List recent events for an account",
        params={"account_id": (str, "acct_001"), "limit": (int, 10)},
        response=[{"account_id": "str", "event": "str", "count": "int"}],
        method="GET",
    )
    def list_events(self, account_id: str, limit: int = 10):
        events = [
            {"account_id": account_id, "event": "login", "count": 48},
            {"account_id": account_id, "event": "export", "count": 7},
            {"account_id": account_id, "event": "invite", "count": 3},
        ]
        return events[:limit]
