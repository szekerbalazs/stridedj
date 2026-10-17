"""Base model and typed identifiers shared by all domain contracts."""

from __future__ import annotations

from typing import NewType

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Base for every domain object: unknown fields are rejected and instances are immutable.

    `extra="forbid"` is also a privacy control: when an external API response is parsed into one of these models,
    any field we did not explicitly allow (GPS, names, ...) raises instead of being silently carried along. Adapters
    therefore pick the allowlisted fields explicitly before constructing a model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


UserId = NewType("UserId", str)
ActivityId = NewType("ActivityId", str)
TrackId = NewType("TrackId", str)
ArtistId = NewType("ArtistId", str)
DraftId = NewType("DraftId", str)
PlanId = NewType("PlanId", str)
SlotId = NewType("SlotId", str)
