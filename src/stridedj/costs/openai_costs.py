"""OpenAI's official billing numbers per API key (Costs API, needs an *Admin* key).

`GET /v1/organization/costs` returns daily buckets of billed amounts; with `group_by=["api_key_id", "line_item"]`
each result is "this key spent X on this line item (e.g. 'gpt-6-luna, input') on this day". Key names come from the
projects' API-key listings. Only `stridedj costs` uses this module: the Admin key can manage the organisation, so it
never enters the long-running bot process (least privilege).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from stridedj.errors import AuthError, ExternalAPIError

MAX_BUCKETS_PER_PAGE = 31


class DailyKeyCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    api_key_id: str
    line_item: str
    amount_usd: float


class ApiKeyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key_id: str
    name: str
    project_name: str
    redacted_value: str  # e.g. "sk-proj-***...abcd" (OpenAI's own masking)


class OfficialCosts(BaseModel):
    model_config = ConfigDict(frozen=True)

    costs: list[DailyKeyCost]
    keys: dict[str, ApiKeyInfo]
    fetched_at: datetime


def redacted_matches(redacted: str, api_key: str) -> bool:
    """Does OpenAI's masked key (visible prefix + visible suffix) belong to `api_key`? Used to link the ledger's
    fingerprint to an official key id without ever sending or storing the key."""
    if "*" not in redacted and "..." not in redacted:
        return False
    head = redacted.split("*")[0].split("...")[0]
    tail = redacted.replace("...", "*").split("*")[-1]
    return bool(head) and bool(tail) and api_key.startswith(head) and api_key.endswith(tail)


def _parse_buckets(pages: list[Any]) -> list[DailyKeyCost]:
    out: list[DailyKeyCost] = []
    for page in pages:
        for bucket in page.data:
            day = datetime.fromtimestamp(bucket.start_time, tz=UTC).date()
            for result in bucket.results:
                amount = getattr(result, "amount", None)
                value = getattr(amount, "value", None) if amount is not None else None
                if value is None:
                    continue
                out.append(
                    DailyKeyCost(
                        day=day,
                        api_key_id=getattr(result, "api_key_id", None) or "unattributed",
                        line_item=getattr(result, "line_item", None) or "other",
                        amount_usd=float(value),
                    )
                )
    return out


async def _key_names(client: AsyncOpenAI) -> dict[str, ApiKeyInfo]:
    keys: dict[str, ApiKeyInfo] = {}
    async for project in client.admin.organization.projects.list(limit=100):
        async for key in client.admin.organization.projects.api_keys.list(project.id, limit=100):
            keys[key.id] = ApiKeyInfo(
                api_key_id=key.id,
                name=key.name,
                project_name=project.name or project.id,
                redacted_value=key.redacted_value,
            )
    return keys


async def fetch_official_costs(admin_key: SecretStr, *, days: int, now: datetime | None = None) -> OfficialCosts:
    """Daily costs per API key and line item for the last `days` days, plus the key names."""
    now = now or datetime.now(UTC)
    start = int((now - timedelta(days=days)).timestamp())
    client = AsyncOpenAI(api_key="unused", admin_api_key=admin_key.get_secret_value(), max_retries=2)
    try:
        pages: list[Any] = []
        cursor: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "start_time": start,
                "bucket_width": "1d",
                "group_by": ["api_key_id", "line_item"],
                "limit": MAX_BUCKETS_PER_PAGE,
            }
            if cursor:
                kwargs["page"] = cursor
            page = await client.admin.organization.usage.costs(**kwargs)
            pages.append(page)
            if not page.has_more or not page.next_page:
                break
            cursor = page.next_page
        keys = await _key_names(client)
    except openai.AuthenticationError:
        raise AuthError("openai-admin", "admin key rejected", status_code=401) from None
    except openai.PermissionDeniedError:
        raise AuthError("openai-admin", "admin key lacks permission for usage/costs", status_code=403) from None
    except openai.APIStatusError as exc:
        raise ExternalAPIError(
            "openai-admin", str(exc.message).splitlines()[0][:200], status_code=exc.status_code
        ) from exc
    except openai.APIConnectionError as exc:
        raise ExternalAPIError("openai-admin", "connection failed", retryable=True) from exc
    finally:
        await client.close()
    return OfficialCosts(costs=_parse_buckets(pages), keys=keys, fetched_at=now)
