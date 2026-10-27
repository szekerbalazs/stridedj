"""Free-form genre tags → the closed `Genre` vocabulary.

Why a closed vocabulary: Last.fm tags are written by any Last.fm user ("seen live", "favorites", or a 200-character
sentence), so they are untrusted text. Mapping them onto ~18 fixed values makes them safe to show to the model (playlist
summaries), usable as a tool enum (the runner's genre choice) and comparable across sources (Last.fm, Spotify, the demo
world). A tag that maps to nothing is dropped.

Matching: the normalised tag is looked up exactly ("deutschrap" → hip hop); otherwise the longest known synonym that
occurs in it as whole words decides ("melodic deep house" → house, "swedish house" → house); otherwise nothing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from stridedj.domain import Genre

MAX_GENRES_PER_ARTIST = 3

_SYNONYMS: dict[Genre, tuple[str, ...]] = {
    Genre.TECHNO: (
        "techno",
        "minimal techno",
        "hard techno",
        "hardtechno",
        "melodic techno",
        "detroit techno",
        "industrial techno",
        "acid techno",
        "peak time techno",
        "tekno",
        "minimal",
    ),
    Genre.HOUSE: (
        "house",
        "deep house",
        "tech house",
        "progressive house",
        "electro house",
        "future house",
        "bass house",
        "slap house",
        "tropical house",
        "afro house",
        "melodic house",
        "french house",
        "chicago house",
        "uk garage",
        "garage",
        "funky house",
    ),
    Genre.EDM: (
        "edm",
        "dance",
        "electronic dance music",
        "dance music",
        "big room",
        "bigroom",
        "electro",
        "eurodance",
        "hardstyle",
        "hard dance",
        "future bass",
        "festival",
    ),
    Genre.TRANCE: (
        "trance",
        "psytrance",
        "psychedelic trance",
        "progressive trance",
        "uplifting trance",
        "goa trance",
        "vocal trance",
    ),
    Genre.DRUM_AND_BASS: (
        "drum and bass",
        "drum n bass",
        "drumnbass",
        "dnb",
        "jungle",
        "liquid funk",
        "liquid drum and bass",
        "neurofunk",
    ),
    Genre.DUBSTEP: ("dubstep", "brostep", "riddim", "bass music"),
    Genre.ELECTRONIC: ("electronic", "electronica", "idm", "synthwave", "retrowave", "electroclash"),
    Genre.HIP_HOP: (
        "hip hop",
        "hiphop",
        "rap",
        "trap",
        "gangsta rap",
        "german hip hop",
        "deutschrap",
        "deutsch rap",
        "drill",
        "uk drill",
        "grime",
        "cloud rap",
        "emo rap",
        "conscious hip hop",
        "underground hip hop",
        "alternative hip hop",
        "east coast rap",
        "west coast rap",
        "southern hip hop",
        "pop rap",
    ),
    Genre.POP: (
        "pop",
        "dance pop",
        "electropop",
        "synthpop",
        "synth pop",
        "teen pop",
        "k pop",
        "kpop",
        "europop",
        "deutschpop",
        "german pop",
        "art pop",
    ),
    Genre.RNB: ("r&b", "rnb", "r and b", "rhythm and blues", "soul", "neo soul", "contemporary r&b", "alternative r&b"),
    Genre.ROCK: (
        "rock",
        "alternative rock",
        "hard rock",
        "classic rock",
        "alternative",
        "rock n roll",
        "grunge",
        "post grunge",
        "pop rock",
        "stoner rock",
    ),
    Genre.INDIE: ("indie", "indie pop", "indie rock", "indie folk", "indietronica"),
    Genre.METAL: (
        "metal",
        "heavy metal",
        "metalcore",
        "nu metal",
        "thrash metal",
        "death metal",
        "alternative metal",
        "deathcore",
        "power metal",
        "black metal",
        "industrial metal",
    ),
    Genre.PUNK: ("punk", "pop punk", "punk rock", "hardcore punk", "post punk", "skate punk", "emo"),
    Genre.LATIN: (
        "latin",
        "reggaeton",
        "latin pop",
        "latin trap",
        "sertanejo",
        "brazilian",
        "salsa",
        "bachata",
        "cumbia",
        "baile funk",
        "funk carioca",
    ),
    Genre.FUNK_DISCO: ("funk", "disco", "nu disco", "boogie"),
    Genre.REGGAE: ("reggae", "dancehall", "ska", "dub"),
    Genre.COUNTRY: ("country", "country pop", "americana", "bluegrass"),
}
_EXACT: dict[str, Genre] = {synonym: genre for genre, synonyms in _SYNONYMS.items() for synonym in synonyms}
_BY_LENGTH: list[tuple[re.Pattern[str], Genre]] = [
    (re.compile(rf"(?<![a-z0-9&]){re.escape(synonym)}(?![a-z0-9&])"), _EXACT[synonym])
    for synonym in sorted(_EXACT, key=lambda s: (-len(s), s))
]
_SEPARATORS = re.compile(r"[-_/.'’]+")
_DISALLOWED = re.compile(r"[^a-z0-9& ]")


def normalise_tag(raw: str) -> str:
    """ "Hip-Hop" → "hip hop", "Drum'n'Bass" → "drumnbass"; ASCII lowercase, single spaces, ≤ 64 chars."""
    ascii_only = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii").lower()
    ascii_only = _SEPARATORS.sub(lambda m: "" if set(m.group()) <= {"'", "’"} else " ", ascii_only)
    return " ".join(_DISALLOWED.sub(" ", ascii_only).split())[:64]


def genre_of_tag(raw: str) -> Genre | None:
    tag = normalise_tag(raw)
    if not tag:
        return None
    if tag in _EXACT:
        return _EXACT[tag]
    for pattern, genre in _BY_LENGTH:
        if pattern.search(tag):
            return genre
    return None


def genres_from_tags(tags: Iterable[str], *, limit: int = MAX_GENRES_PER_ARTIST) -> tuple[Genre, ...]:
    """The genres of an artist from its tags (most important tag first), de-duplicated, at most `limit`."""
    found: dict[Genre, None] = {}
    for tag in tags:
        genre = genre_of_tag(tag)
        if genre is not None:
            found[genre] = None
        if len(found) >= limit:
            break
    return tuple(found)
