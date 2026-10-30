"""Render explicit Ukrainian target profiles without authoring dataset sentences."""

from fairforget.warbias.profiles import SUPPORTED_ATTRIBUTES

OCCUPATION_SECTORS_UK = {
    "administration": "офісної або адміністративної роботи",
    "agriculture": "сільського господарства, лісового господарства або садівництва",
    "education": "освіти або навчання",
    "engineering": "інженерії, науки або технічної роботи",
    "finance_legal": "фінансів, бізнесу або права",
    "healthcare": "охорони здоров’я або оплачуваного догляду",
    "hospitality": "готельних послуг або громадського харчування",
    "it": "інформаційних технологій",
    "manufacturing": "виробництва або ремонту",
    "media": "медіа, мистецтва або комунікацій",
    "property_services": "прибирання або обслуговування нерухомості",
    "retail": "роздрібної торгівлі, продажів або обслуговування клієнтів",
    "trades": "будівництва або робітничих ремесел",
    "transport": "транспорту або логістики",
}

ABROAD_SUBGROUPS_UK = {
    "abroad_refugee_status": "дорослі люди з України зі статусом біженця за кордоном",
    "abroad_temporary_protection": "дорослі люди з України під тимчасовим захистом за кордоном",
}

REPORTERS_UK = {
    "another veteran": "інший ветеран",
    "business adviser": "консультант із ведення бізнесу",
    "career adviser": "кар’єрний консультант",
    "caseworker": "фахівець із супроводу",
    "caseworker in the host community": "фахівець із супроводу в громаді країни перебування",
    "colleague": "колега",
    "community member in Ukraine": "член громади в Україні",
    "community organiser": "організатор громадської діяльності",
    "community organiser in the host community": (
        "організатор громадської діяльності в громаді країни перебування"
    ),
    "hiring manager": "менеджер із найму",
    "journalist": "журналіст",
    "landlord": "орендодавець",
    "letting agent": "агент з оренди житла",
    "local resident": "місцевий мешканець",
    "local resident in the host community": "місцевий мешканець громади країни перебування",
    "neighbour": "сусід",
    "neighbour in Ukraine": "сусід в Україні",
    "news editor": "редактор новин",
    "recruiter": "рекрутер",
    "rehabilitation professional": "фахівець із реабілітації",
    "service clerk": "службовець із прийому звернень",
    "supervisor": "керівник",
}


def render_reporter(role):
    """Return the selected speaker role without choosing a speaker for the case."""
    if role is None:
        return None
    if not isinstance(role, str) or role not in REPORTERS_UK:
        raise ValueError("Unknown reporter role")
    return REPORTERS_UK[role]


MODIFIERS = {
    "age": {
        "young": "молодого віку",
        "middle_aged": "середнього віку",
        "older": "старшого віку",
    },
    "ethnicity": {"roma": "ромського походження"},
    "disability": {
        "unspecified_cause": "з інвалідністю",
        "war_caused": "з інвалідністю внаслідок війни",
        "war_caused_nonvisible": "з невидимою інвалідністю внаслідок війни",
    },
    "origin_region": {"east": "зі сходу України"},
    "pets": {"with_pets": "з домашніми тваринами"},
}

PREDICATES = {
    "service_period": {
        "ato_oos": "служили під час антитерористичної операції або операції Об’єднаних сил",
        "full_scale": "служили під час повномасштабної війни",
        "both": (
            "служили під час антитерористичної операції або операції Об’єднаних сил, "
            "а також під час повномасштабної війни"
        ),
    },
    "service_role": {
        "combat": "мають досвід бойової служби",
        "noncombat": "мають досвід небойової військової служби",
    },
    "rank": {
        "enlisted": "мають військові звання рядового складу",
        "sergeant": "мають військові звання сержантського складу",
        "officer": "мають офіцерські військові звання",
    },
    "recruitment_route": {
        "contract": "вступили на військову службу за контрактом",
        "mobilized": "були мобілізовані",
        "volunteer": "добровільно вступили на військову службу",
    },
    "service_duration": {
        "long_service": "мають тривалий досвід військової служби",
        "short_service": "мають нетривалий досвід військової служби",
    },
    "service_state": {
        "currently_serving": "зараз проходять військову службу",
        "discharged": "звільнені з військової служби",
        "returned_to_service": "повернулися на військову службу",
    },
    "discharge_recency": {
        "recently_discharged": "нещодавно були звільнені з військової служби",
        "longer_civilian_life": (
            "вже тривалий час живуть цивільним життям після звільнення з військової служби"
        ),
    },
    "displacement_origin": {"occupied_territories": "виїхали з окупованих територій України"},
    "displacement_wave": {
        "before_full_scale": "були вимушено переміщені до початку повномасштабного вторгнення",
        "during_full_scale": "були вимушено переміщені під час повномасштабної війни",
    },
    "registration": {"unregistered": "не зареєстровані як внутрішньо переміщені особи"},
    "language": {
        "ukrainian": "спілкуються українською мовою",
        "russian": "спілкуються російською мовою",
    },
    "housing_type": {
        "collective_accommodation": "проживають у місцях колективного проживання",
        "rented_home": "проживають в орендованому житлі",
    },
    "caregiving": {
        "parent": "мають дітей",
        "unpaid_carer": "безоплатно доглядають за іншими людьми",
    },
    "help_seeking": {"psychological_support": "звертаються по психологічну підтримку"},
}

SPECIAL_VALUES = {
    "gender": {"men", "women"},
    "dual_status": {"veteran_and_idp"},
    "region": {"eastern_region", "western_region"},
    "community_type": {"rural", "urban"},
    "occupation": set(OCCUPATION_SECTORS_UK),
}


def _attributes(value):
    if not isinstance(value, dict) or not set(value) <= SUPPORTED_ATTRIBUTES - {"prior_occupation"}:
        raise ValueError("Unknown target profile attribute")
    result = dict(value)
    if result.get("age") == "young_adult":
        result["age"] = "young"
    occupation = result.get("occupation")
    if isinstance(occupation, str) and occupation.startswith("family_"):
        result["occupation"] = occupation.removeprefix("family_")
    for axis, item in result.items():
        allowed = MODIFIERS.get(axis, PREDICATES.get(axis, SPECIAL_VALUES.get(axis, {})))
        if not isinstance(item, str) or item not in allowed:
            raise ValueError(f"Unknown target profile value for {axis}")
    return result


def _base(case, gender, subgroup_uk):
    group = case.get("target_group")
    if group == "veteran":
        return {
            None: "Дорослі ветерани війни",
            "men": "Дорослі чоловіки-ветерани війни",
            "women": "Дорослі жінки-ветеранки війни",
        }[gender]
    if group == "idp":
        return (
            "Дорослі внутрішньо переміщені "
            + {
                None: "особи",
                "men": "чоловіки",
                "women": "жінки",
            }[gender]
        )
    if group != "displaced_abroad":
        raise ValueError("Unknown target group")
    if not isinstance(subgroup_uk, str):
        raise ValueError("Abroad targets need an explicit supported Ukrainian subgroup")
    canonical = next(
        (
            text
            for text in ABROAD_SUBGROUPS_UK.values()
            if text.casefold() == (subgroup_uk or "").casefold()
        ),
        None,
    )
    if canonical is None:
        raise ValueError("Abroad targets need an explicit supported Ukrainian subgroup")
    code = case.get("target_code")
    if code is not None and ABROAD_SUBGROUPS_UK.get(code) != canonical:
        raise ValueError("Abroad target code and Ukrainian subgroup differ")
    person = {None: "люди", "men": "чоловіки", "women": "жінки"}[gender]
    text = canonical.replace("дорослі люди", "дорослі " + person, 1)
    return text[0].upper() + text[1:]


def render_target(case, attributes, occupation_reference=None, target_subgroup_uk=None):
    """Return an adult plural noun phrase with every declared attribute retained."""
    attributes = _attributes(attributes)
    intrinsic = _attributes(case.get("intrinsic_attributes", {}))
    if any(attributes.get(axis) != value for axis, value in intrinsic.items()):
        raise ValueError("Target wording must preserve every intrinsic attribute")
    if "occupation" in attributes:
        if occupation_reference not in {"current", "previous", "sought"}:
            raise ValueError("Occupation wording needs an explicit temporal reference")
    elif occupation_reference is not None:
        raise ValueError("An occupation reference needs an occupation attribute")
    group = case.get("target_group")
    gender = attributes.get("gender")
    target = _base(case, gender, target_subgroup_uk or case.get("target_subgroup_uk"))
    modifiers = [
        values[attributes[axis]] for axis, values in MODIFIERS.items() if axis in attributes
    ]
    if modifiers:
        target += " " + " ".join(modifiers)
    predicates = [
        values[attributes[axis]] for axis, values in PREDICATES.items() if axis in attributes
    ]
    if "dual_status" in attributes:
        if group == "veteran":
            predicates.append("також є внутрішньо переміщеними особами")
        elif group == "idp":
            predicates.append(
                "також є " + ("ветеранками війни" if gender == "women" else "ветеранами війни")
            )
        else:
            raise ValueError("Veteran and IDP dual status needs one of those target groups")
    if "region" in attributes:
        direction = {"eastern_region": "східному", "western_region": "західному"}[
            attributes["region"]
        ]
        country = "країни перебування" if group == "displaced_abroad" else "України"
        predicates.append(f"проживають у {direction} регіоні {country}")
    if "community_type" in attributes:
        place = {"rural": "у сільській місцевості", "urban": "у місті"}[
            attributes["community_type"]
        ]
        if group == "displaced_abroad":
            place += " країни перебування"
        predicates.append("проживають " + place)
    if "occupation" in attributes:
        work = {
            "current": "працюють",
            "previous": "раніше працювали",
            "sought": "шукають роботу",
        }[occupation_reference]
        predicates.append(f"{work} у сфері {OCCUPATION_SECTORS_UK[attributes['occupation']]}")
    if predicates:
        target += ", які " + " та ".join(predicates)
    return target
