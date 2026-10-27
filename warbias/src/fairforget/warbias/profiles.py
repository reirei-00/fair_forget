"""Natural English profile realization; outputs remain subject to direct review.

The original profile renderer concatenated taxonomy labels into long noun phrases
(``female veterans``, ``disabilities caused by war``, and repeated ``who`` clauses).
This revision keeps every selected attribute explicit while assembling the profile as
ordinary English.  Historical outputs are preserved in the run directory; this module
is used for subsequent wording revisions.
"""

AGE_WORDS = {"young_adult": "young", "middle_aged": "middle-aged", "older": "older"}
GENDER_WORDS = {"women": "women", "men": "men"}
SERVICE_PERIOD_CLAUSES = {
    "ato_oos": "who served during ATO/OOS (2014–2021)",
    "full_scale": "who served during the full-scale war",
    "both": "who served both before and during the full-scale invasion",
}
ORIGIN_WORDS = {"occupied_territories": "occupied territories"}
ORIGIN_REGION_WORDS = {"east": "eastern regions"}
DISABILITY_MODIFIERS = {
    "war_caused": "with war-related disabilities",
    "war_caused_nonvisible": "with invisible war-related disabilities",
    "unspecified_cause": "with disabilities",
}
PETS_MODIFIERS = {"with_pets": "with pets"}
REGISTRATION_CLAUSES = {"unregistered": "who lack formal IDP registration"}
DUAL_STATUS_CLAUSES = {
    "veteran_and_idp": {
        "veteran": "who are internally displaced",
        "idp": "who are veterans",
    }
}
LANGUAGE_PREFIXES = {"russian": "Russian-speaking ", "ukrainian": "Ukrainian-speaking "}
ETHNICITY_PREFIXES = {"roma": "Roma "}
SUFFIX_CLAUSES = {
    "service_state": {
        "discharged": "who have left military service",
        "currently_serving": "who are currently serving",
        "returned_to_service": "who have returned to military service",
    },
    "discharge_recency": {
        "recently_discharged": "who recently left military service",
        "longer_civilian_life": "who have been living as civilians for some time",
    },
    "service_duration": {
        "short_service": "who served for a relatively short time",
        "long_service": "who have a long history of military service",
    },
    "service_role": {
        "combat": "who served in combat roles",
        "noncombat": "who served in noncombat roles",
    },
    "rank": {
        "officer": "who served as officers",
        "sergeant": "who served as sergeants",
        "enlisted": "who served in the enlisted ranks",
    },
    "recruitment_route": {
        "volunteer": "who volunteered for military service",
        "contract": "who entered military service on a contract",
        "mobilized": "who entered military service through mobilization",
    },
    "prior_occupation": {
        "schoolteacher": "who previously worked as schoolteachers",
        "electrician": "who previously worked as electricians",
    },
    "region": {
        "eastern_region": "who live in an eastern region",
        "western_region": "who live in a western region",
    },
    "community_type": {
        "urban": "who live in an urban community",
        "rural": "who live in a rural community",
    },
    "displacement_wave": {
        "before_full_scale": "who were displaced before the full-scale invasion",
        "during_full_scale": "who were displaced during the full-scale invasion",
    },
    "housing_type": {
        "rented_home": "who live in rented housing",
        "collective_accommodation": "who live in collective accommodation",
    },
    "caregiving": {
        "parent": "who have children",
        "unpaid_carer": "who provide unpaid care for a relative",
    },
    "help_seeking": {"psychological_support": "who seek psychological support"},
}
ATTRIBUTE_WORDS = {
    "age": AGE_WORDS,
    "gender": GENDER_WORDS,
    "service_period": SERVICE_PERIOD_CLAUSES,
    "displacement_origin": ORIGIN_WORDS,
    "origin_region": ORIGIN_REGION_WORDS,
    "disability": DISABILITY_MODIFIERS,
    "pets": PETS_MODIFIERS,
    "registration": REGISTRATION_CLAUSES,
    "dual_status": DUAL_STATUS_CLAUSES,
    "language": LANGUAGE_PREFIXES,
    "ethnicity": ETHNICITY_PREFIXES,
    **SUFFIX_CLAUSES,
}
SUPPORTED_ATTRIBUTES = {*ATTRIBUTE_WORDS, "occupation"}
ATTRIBUTE_TARGET_GROUPS = {
    "service_period": "veteran",
    "displacement_origin": "idp",
    "origin_region": "idp",
}
OCCUPATION_SECTORS = {
    "family_education": "education",
    "family_healthcare": "healthcare and care services",
    "family_it": "information technology",
    "family_engineering": "engineering and technical occupations",
    "family_administration": "administration",
    "family_finance_legal": "finance and legal services",
    "family_retail": "retail and customer service",
    "family_hospitality": "hospitality and food services",
    "family_trades": "construction and the skilled trades",
    "family_manufacturing": "manufacturing and repair services",
    "family_transport": "transport and logistics",
    "family_agriculture": "agriculture and forestry",
    "family_media": "media and the arts",
    "family_property_services": "cleaning and property services",
}
OCCUPATION_CLAUSES = {
    "current": "who work ",
    "previous": "who previously worked ",
    "sought": "who are seeking work ",
}


def _require_option(axis, value, vocabulary):
    if not isinstance(value, str) or value not in vocabulary:
        raise ValueError(f"Unsupported {axis} value {value!r} for profile rendering")


def _validate_profile(row, attributes):
    if row.get("target_group") not in ("idp", "veteran"):
        raise ValueError("Unsupported target group for profile rendering")
    if set(attributes) - SUPPORTED_ATTRIBUTES:
        raise ValueError("Profile includes attributes this renderer cannot realize")
    for axis, vocabulary in ATTRIBUTE_WORDS.items():
        value = attributes.get(axis)
        if value is None or value == "":
            continue
        _require_option(axis, value, vocabulary)
        target_group = ATTRIBUTE_TARGET_GROUPS.get(axis)
        if target_group and row["target_group"] != target_group:
            raise ValueError(
                f"Profile renderer cannot realize {axis} for target group {row['target_group']!r}"
            )
    occupation = attributes.get("occupation")
    if occupation is not None and occupation != "" and not isinstance(occupation, str):
        raise ValueError(f"Unsupported occupation value {occupation!r} for profile rendering")


def plural(job):
    return (
        job + "es"
        if job.endswith(("ch", "sh", "s", "x"))
        else job[:-1] + "ies"
        if job.endswith("y") and job[-2] not in "aeiou"
        else job + "s"
    )


def _join_clauses(base, clauses):
    """Attach relative clauses without producing ``who ... and who ...`` stacks."""
    if not clauses:
        return base
    rendered = []
    for clause in clauses:
        if clause.startswith("who "):
            clause = clause[4:]
        rendered.append(clause)
    return base + " who " + rendered[0] + "".join(" and " + c for c in rendered[1:])


def _occupation_clause(occupation, reference, dictionary):
    _require_option("occupation_reference", reference, OCCUPATION_CLAUSES)
    families = {x["code"] for x in dictionary.get("occupations", [])}
    jobs = {j["code"]: j["label_en"] for x in dictionary.get("occupations", []) for j in x["jobs"]}
    if not isinstance(occupation, str) or occupation not in families | jobs.keys():
        raise ValueError(
            f"Unsupported occupation value {occupation!r} in the provided attribute dictionary"
        )
    if occupation in families:
        _require_option("occupation family", occupation, OCCUPATION_SECTORS)
        noun = "in " + OCCUPATION_SECTORS[occupation]
    else:
        noun = "as " + plural(jobs[occupation])
    return OCCUPATION_CLAUSES[reference] + noun


def target(row, dictionary=None):
    """Render a profile as natural English while preserving every selected attribute."""
    attrs = {**row["intrinsic_attributes"], **row["additional_tags"]}
    _validate_profile(row, attrs)
    age = AGE_WORDS.get(attrs.get("age"), "")
    modifiers = []
    clauses = []

    if row["target_group"] == "veteran":
        # ``women veterans`` is the ordinary English label; pair it with the
        # equally idiomatic ``male veterans`` rather than ``men veterans``.
        gender = {"women": "women", "men": "male"}.get(attrs.get("gender"), "")
        base = " ".join(x for x in (age, gender, "veterans") if x)
        period = attrs.get("service_period")
        if period:
            clauses.append(SERVICE_PERIOD_CLAUSES[period])
    else:
        person = GENDER_WORDS.get(attrs.get("gender"), "people")
        base = " ".join(x for x in (age, "internally displaced", person) if x)
        origin = attrs.get("displacement_origin")
        region = attrs.get("origin_region")
        if origin == "occupied_territories" and region == "east":
            base += " from eastern occupied territories"
        elif origin:
            base += " from " + ORIGIN_WORDS[origin]
        elif region:
            base += " from " + ORIGIN_REGION_WORDS[region]

    disability = attrs.get("disability")
    if disability:
        modifiers.append(DISABILITY_MODIFIERS[disability])

    if attrs.get("pets") and not attrs.get("occupation"):
        modifiers.append(PETS_MODIFIERS[attrs["pets"]])
    if attrs.get("registration"):
        clauses.append(REGISTRATION_CLAUSES[attrs["registration"]])

    occupation = attrs.get("occupation")
    if occupation:
        if dictionary is None:
            raise ValueError("An occupation profile requires an explicit attribute dictionary")
        clauses.append(
            _occupation_clause(
                occupation,
                row.get("occupation_reference", "current"),
                dictionary,
            )
        )
        if attrs.get("pets") == "with_pets":
            clauses.append("who have pets")

    for axis, options in SUFFIX_CLAUSES.items():
        if attrs.get(axis):
            clauses.append(options[attrs[axis]])

    if attrs.get("dual_status"):
        clauses.append(DUAL_STATUS_CLAUSES[attrs["dual_status"]][row["target_group"]])

    if modifiers:
        base += " " + " and ".join(modifiers)
    base = _join_clauses(base, clauses)
    if attrs.get("language"):
        base = LANGUAGE_PREFIXES[attrs["language"]] + base
    if attrs.get("ethnicity"):
        base = ETHNICITY_PREFIXES[attrs["ethnicity"]] + base
    return base


def frame(text, reporter):
    if not reporter:
        return text[0].upper() + text[1:]
    prefix = (
        "Another veteran" if reporter == "another veteran" else "A " + reporter
    ) + " says that "
    # Attribution changes sentence position, not proper-name capitalization.
    if text.startswith(("Roma ", "Russian-speaking ", "Ukrainian-speaking ")):
        return prefix + text
    return prefix + text[0].lower() + text[1:]


def build_profile_matrix(cases, profiles, allocations):
    """Resolve an explicit frozen allocation without inventing eligible combinations."""
    from fairforget.io import digest, unique_index

    cases = unique_index(cases, "case_id")
    profiles = unique_index(profiles, "profile_id")
    unique_index(allocations)
    output = []
    for allocation in allocations:
        case = cases[allocation["case_id"]]
        profile = profiles[allocation["profile_id"]]
        intrinsic = dict(case.get("intrinsic_attributes", {}))
        additional = dict(profile.get("additional_tags", {}))
        if set(intrinsic) & set(additional):
            raise ValueError("Profile attributes cannot overwrite a case's intrinsic attributes")
        if profile.get("target_group", case["target_group"]) != case["target_group"]:
            raise ValueError("Profile target group differs from the source case")
        row = {
            **allocation,
            "target_group": case["target_group"],
            "claim_family_id": case["claim_family_id"],
            "case_spec_sha256": case["case_spec_sha256"],
            "intrinsic_attributes": intrinsic,
            "additional_tags": additional,
            "profile_sha256": digest(profile),
            "human_validated": False,
        }
        row["allocation_sha256"] = digest(allocation)
        output.append(row)
    return output
