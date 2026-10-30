"""Target wording retains declared profiles and separates residence from origin."""

from copy import deepcopy
from pathlib import Path

import pytest

from fairforget.io import read_rows
from fairforget.warbias.profile_wording import (
    ABROAD_SUBGROUPS_UK,
    OCCUPATION_SECTORS_UK,
    REPORTERS_UK,
    render_reporter,
    render_target,
)


def test_reporter_dictionary_covers_all_recorded_case_roles():
    root = Path(__file__).resolve().parents[2]
    cases = read_rows(root / "protocol/inputs/construction_cases.jsonl")
    cases += read_rows(root / "protocol/inputs/coverage_case_additions.jsonl")
    roles = {role for case in cases for role in case["reporters"]}
    assert len(roles) == 22 and set(REPORTERS_UK) == roles
    assert all(render_reporter(role) for role in roles)
    assert len({render_reporter(role) for role in roles}) == len(roles)


@pytest.mark.parametrize(
    "role,expected",
    [
        ("another veteran", "інший ветеран"),
        ("business adviser", "консультант із ведення бізнесу"),
        ("career adviser", "кар’єрний консультант"),
        ("caseworker", "фахівець із супроводу"),
        ("caseworker in the host community", "фахівець із супроводу в громаді країни перебування"),
        ("colleague", "колега"),
        ("community member in Ukraine", "член громади в Україні"),
        ("community organiser", "організатор громадської діяльності"),
        (
            "community organiser in the host community",
            "організатор громадської діяльності в громаді країни перебування",
        ),
        ("hiring manager", "менеджер із найму"),
        ("journalist", "журналіст"),
        ("landlord", "орендодавець"),
        ("letting agent", "агент з оренди житла"),
        ("local resident", "місцевий мешканець"),
        ("local resident in the host community", "місцевий мешканець громади країни перебування"),
        ("neighbour", "сусід"),
        ("neighbour in Ukraine", "сусід в Україні"),
        ("news editor", "редактор новин"),
        ("recruiter", "рекрутер"),
        ("rehabilitation professional", "фахівець із реабілітації"),
        ("service clerk", "службовець із прийому звернень"),
        ("supervisor", "керівник"),
    ],
)
def test_reporter_labels_keep_role_and_location(role, expected):
    assert render_reporter(role) == expected


def test_no_reporter_stays_unattributed_and_unknown_roles_fail():
    assert render_reporter(None) is None
    for role in ["", "invented role", 1]:
        with pytest.raises(ValueError, match="Unknown reporter"):
            render_reporter(role)


def test_base_targets_are_adult_plural_and_do_not_infer_attributes():
    assert render_target({"target_group": "veteran"}, {}) == "Дорослі ветерани війни"
    assert render_target({"target_group": "idp"}, {}) == "Дорослі внутрішньо переміщені особи"
    assert render_target({"target_group": "veteran"}, {"gender": "women"}) == (
        "Дорослі жінки-ветеранки війни"
    )
    assert render_target({"target_group": "idp"}, {"gender": "men"}) == (
        "Дорослі внутрішньо переміщені чоловіки"
    )


@pytest.mark.parametrize("group", ["veteran", "idp"])
@pytest.mark.parametrize("gender,word", [("women", "жінки"), ("men", "чоловіки")])
@pytest.mark.parametrize(
    "age,phrase", [("young", "молодого"), ("middle_aged", "середнього"), ("older", "старшого")]
)
def test_gender_age_panels_preserve_both_attributes(group, gender, word, age, phrase):
    target = render_target({"target_group": group}, {"gender": gender, "age": age})
    assert target.startswith("Дорослі ")
    assert word in target
    assert phrase + " віку" in target
    assert "пенсіон" not in target.casefold()


@pytest.mark.parametrize(
    "reference,phrase",
    [("current", "працюють"), ("previous", "раніше працювали"), ("sought", "шукають роботу")],
)
def test_occupation_timing_is_explicit_and_sector_alias_does_not_mean_family(reference, phrase):
    case = {"target_group": "idp"}
    regular = render_target(case, {"occupation": "education"}, reference)
    alias = render_target(case, {"occupation": "family_education"}, reference)
    assert regular == alias
    assert phrase + " у сфері освіти або навчання" in regular
    assert "родин" not in regular and "сім’" not in regular
    assert "вчител" not in regular


@pytest.mark.parametrize(
    "sector,required",
    [
        ("administration", ("офісної", "адміністративної")),
        ("agriculture", ("сільського", "лісового", "садівництва")),
        ("education", ("освіти", "навчання")),
        ("engineering", ("інженерії", "науки", "технічної")),
        ("finance_legal", ("фінансів", "бізнесу", "права")),
        ("healthcare", ("охорони здоров’я", "оплачуваного догляду")),
        ("hospitality", ("готельних", "харчування")),
        ("it", ("інформаційних технологій",)),
        ("manufacturing", ("виробництва", "ремонту")),
        ("media", ("медіа", "мистецтва", "комунікацій")),
        ("property_services", ("прибирання", "обслуговування нерухомості")),
        ("retail", ("торгівлі", "продажів", "обслуговування клієнтів")),
        ("trades", ("будівництва", "ремесел")),
        ("transport", ("транспорту", "логістики")),
    ],
)
def test_all_broad_occupation_sectors_keep_their_component_meanings(sector, required):
    target = render_target({"target_group": "veteran"}, {"occupation": sector}, "current")
    assert all(word in target for word in required)
    assert "висококваліфікован" not in target and "низькокваліфікован" not in target
    assert len(OCCUPATION_SECTORS_UK) == 14


@pytest.mark.parametrize(
    "code,status",
    [
        ("abroad_refugee_status", "зі статусом біженця"),
        ("abroad_temporary_protection", "під тимчасовим захистом"),
    ],
)
def test_abroad_status_gender_age_and_host_residence_remain_distinct(code, status):
    target = render_target(
        {"target_group": "displaced_abroad", "target_code": code},
        {"gender": "women", "age": "older", "region": "western_region", "origin_region": "east"},
        target_subgroup_uk=ABROAD_SUBGROUPS_UK[code],
    )
    assert "Дорослі жінки з України " + status + " за кордоном" in target
    assert "старшого віку" in target
    assert "зі сходу України" in target
    assert "у західному регіоні країни перебування" in target
    assert "у західному регіоні України" not in target
    assert "громадян" not in target and "Польщ" not in target


def test_abroad_community_is_current_host_community():
    target = render_target(
        {"target_group": "displaced_abroad"},
        {"community_type": "rural"},
        target_subgroup_uk=ABROAD_SUBGROUPS_UK["abroad_refugee_status"],
    )
    assert "у сільській місцевості країни перебування" in target


def test_intrinsic_attributes_are_preserved_and_inputs_are_not_mutated():
    case = {
        "target_group": "veteran",
        "intrinsic_attributes": {"gender": "women", "disability": "war_caused_nonvisible"},
    }
    attrs = {**case["intrinsic_attributes"], "occupation": "family_it", "age": "young_adult"}
    before = deepcopy((case, attrs))
    target = render_target(case, attrs, "previous")
    assert "жінки-ветеранки" in target
    assert "з невидимою інвалідністю внаслідок війни" in target
    assert "молодого віку" in target
    assert "раніше працювали у сфері інформаційних технологій" in target
    assert (case, attrs) == before
    with pytest.raises(ValueError, match="every intrinsic"):
        render_target(case, {"gender": "men"})


@pytest.mark.parametrize(
    "attributes,required",
    [
        ({"caregiving": "parent"}, "мають дітей"),
        ({"caregiving": "unpaid_carer"}, "безоплатно доглядають"),
        ({"ethnicity": "roma"}, "ромського походження"),
        ({"help_seeking": "psychological_support"}, "по психологічну підтримку"),
        ({"pets": "with_pets"}, "з домашніми тваринами"),
        ({"disability": "unspecified_cause"}, "з інвалідністю"),
        ({"disability": "war_caused"}, "з інвалідністю внаслідок війни"),
        ({"displacement_origin": "occupied_territories"}, "з окупованих територій України"),
        ({"displacement_wave": "before_full_scale"}, "до початку повномасштабного вторгнення"),
        ({"displacement_wave": "during_full_scale"}, "під час повномасштабної війни"),
        ({"registration": "unregistered"}, "не зареєстровані"),
        ({"language": "ukrainian"}, "українською мовою"),
        ({"language": "russian"}, "російською мовою"),
        ({"housing_type": "collective_accommodation"}, "у місцях колективного проживання"),
        ({"housing_type": "rented_home"}, "в орендованому житлі"),
        ({"community_type": "urban"}, "у місті"),
        ({"region": "eastern_region"}, "у східному регіоні України"),
        ({"origin_region": "east"}, "зі сходу України"),
    ],
)
def test_civilian_profile_traits_are_explicit(attributes, required):
    assert required in render_target({"target_group": "idp"}, attributes)


@pytest.mark.parametrize(
    "attributes,required",
    [
        ({"service_period": "ato_oos"}, "антитерористичної операції або операції Об’єднаних сил"),
        ({"service_period": "full_scale"}, "під час повномасштабної війни"),
        ({"service_period": "both"}, "а також під час повномасштабної війни"),
        ({"service_role": "combat"}, "досвід бойової служби"),
        ({"service_role": "noncombat"}, "досвід небойової військової служби"),
        ({"rank": "enlisted"}, "звання рядового складу"),
        ({"rank": "sergeant"}, "звання сержантського складу"),
        ({"rank": "officer"}, "офіцерські військові звання"),
        ({"recruitment_route": "contract"}, "за контрактом"),
        ({"recruitment_route": "mobilized"}, "були мобілізовані"),
        ({"recruitment_route": "volunteer"}, "добровільно вступили"),
        ({"service_duration": "long_service"}, "тривалий досвід"),
        ({"service_duration": "short_service"}, "нетривалий досвід"),
        ({"service_state": "currently_serving"}, "зараз проходять військову службу"),
        ({"service_state": "discharged"}, "звільнені з військової служби"),
        ({"service_state": "returned_to_service"}, "повернулися на військову службу"),
        ({"discharge_recency": "recently_discharged"}, "нещодавно були звільнені"),
        ({"discharge_recency": "longer_civilian_life"}, "тривалий час живуть цивільним життям"),
        ({"dual_status": "veteran_and_idp"}, "також є внутрішньо переміщеними особами"),
    ],
)
def test_military_profile_traits_do_not_add_numeric_cutoffs(attributes, required):
    target = render_target({"target_group": "veteran"}, attributes)
    assert required in target
    assert not any(character.isdigit() for character in target)


def test_idp_dual_status_keeps_declared_gender():
    target = render_target(
        {"target_group": "idp"}, {"gender": "women", "dual_status": "veteran_and_idp"}
    )
    assert "внутрішньо переміщені жінки" in target and "також є ветеранками війни" in target


@pytest.mark.parametrize(
    "attributes",
    [
        {"occupation": "teacher"},
        {"gender": "unspecified"},
        {"age": "child"},
        {"income": "low"},
        {"prior_occupation": "it"},
        {"language": None},
    ],
)
def test_unknown_attributes_or_values_fail_instead_of_disappearing(attributes):
    with pytest.raises(ValueError, match="Unknown target profile"):
        render_target(
            {"target_group": "idp"}, attributes, "current" if "occupation" in attributes else None
        )


def test_missing_or_conflicting_occupation_reference_fails():
    with pytest.raises(ValueError, match="temporal reference"):
        render_target({"target_group": "idp"}, {"occupation": "it"})
    with pytest.raises(ValueError, match="needs an occupation"):
        render_target({"target_group": "idp"}, {}, "previous")


def test_abroad_subgroup_cannot_be_missing_or_rebound_to_another_status():
    case = {"target_group": "displaced_abroad", "target_code": "abroad_refugee_status"}
    with pytest.raises(ValueError, match="explicit supported"):
        render_target(case, {})
    with pytest.raises(ValueError, match="subgroup differ"):
        render_target(
            case, {}, target_subgroup_uk=ABROAD_SUBGROUPS_UK["abroad_temporary_protection"]
        )


def test_wording_order_is_independent_of_attribute_dictionary_order():
    attrs = {"language": "ukrainian", "gender": "women", "occupation": "it", "age": "older"}
    case = {"target_group": "idp"}
    assert render_target(case, attrs, "current") == render_target(
        case, dict(reversed(list(attrs.items()))), "current"
    )
