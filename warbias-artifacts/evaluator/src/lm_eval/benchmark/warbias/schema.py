"""Shared vocabulary and the item record every WarBias analysis consumes."""

from dataclasses import asdict, dataclass


LANGUAGES = ("uk", "en")
ACTORS = ("idp", "veteran")
GENDERS = ("men", "women", "not_stated")
AGES = ("young", "middle_aged", "older", "not_stated")
AXES = ("actor", "gender", "age")
# A disclosure condition (gender or age not mentioned), not a sensitive group:
# never compared as a group, reported as a reference on the same cases.
NOT_STATED = "not_stated"
REFERENCE_VALUES = (NOT_STATED,)
SENTENCE_ROLES = ("stereotype", "counter_stereotype", "unrelated")
LENGTH_SCORINGS = ("per_token", "per_byte", "sum")


def validate_language_pair(pair):
    """Return two distinct language codes in their explicit comparison order."""
    if (
        not isinstance(pair, (tuple, list))
        or len(pair) != 2
        or any(not isinstance(code, str) or not code.strip() for code in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError("A language pair requires two distinct nonempty codes")
    return tuple(pair)


@dataclass(frozen=True, slots=True)
class TripletItem:
    """One scored triplet.

    ``preference`` is 1.0 when the stereotype is preferred, 0.0 when the
    counter-stereotype is, and None when there is no answer (abstention or a
    missing score). ``cluster`` is the claim component, the independent
    sampling unit. Scoring-specific fields stay None when not applicable.
    """

    id: str
    cluster: str
    actor: str
    gender: str
    age: str
    comparison_unit: str | None
    claim_family_id: str | None = None
    cross_actor_claim_id: str | None = None
    preference: float | None = None
    tie: float = 0.0
    margin: float | None = None
    delta_stereotype: float | None = None
    delta_counter: float | None = None
    delta_unrelated: float | None = None
    length_difference: int | None = None
    lms_per_token: float | None = None
    lms_per_byte: float | None = None
    lms_sum: float | None = None

    def to_dict(self):
        return asdict(self)


def value(item, field):
    """Read ``field`` from a TripletItem or a mapping (legacy row dicts)."""
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field)
