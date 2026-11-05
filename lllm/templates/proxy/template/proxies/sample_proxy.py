from lllm import BaseProxy


class SampleDataProxy(BaseProxy):
    _proxy_path = "sample"
    _proxy_name = "Sample Data Proxy"
    _proxy_description = "Example local API surface for account and event data."

    # Each endpoint method receives the call's params as one dict, as in
    # CALL_API("sample/events", {"account_id": "acct_002"}). This proxy has no
    # base_url, so the methods compute their responses themselves.

    @BaseProxy.endpoint(
        category="accounts",
        endpoint="/accounts",
        description="List known accounts",
        params={"limit": (int, 10)},
        response=[{"id": "str", "name": "str", "tier": "str"}],
        method="GET",
    )
    def list_accounts(self, params: dict) -> list:
        accounts = [
            {"id": "acct_001", "name": "Northwind", "tier": "enterprise"},
            {"id": "acct_002", "name": "Contoso", "tier": "startup"},
            {"id": "acct_003", "name": "Fabrikam", "tier": "growth"},
        ]
        return accounts[: params.get("limit", 10)]

    @BaseProxy.endpoint(
        category="events",
        endpoint="/events",
        description="List recent events for an account",
        params={"account_id*": (str, "acct_001"), "limit": (int, 10)},
        response=[{"account_id": "str", "event": "str", "count": "int"}],
        method="GET",
    )
    def list_events(self, params: dict) -> list:
        account_id = params["account_id"]
        events = [
            {"account_id": account_id, "event": "login", "count": 48},
            {"account_id": account_id, "event": "export", "count": 7},
            {"account_id": account_id, "event": "invite", "count": 3},
        ]
        return events[: params.get("limit", 10)]
