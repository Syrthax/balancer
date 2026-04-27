import pytest
from balancer.generator import generate_pairs, _verify_identical_profiles

PROFILE_FIELDS = [
    "years_experience",
    "skills",
    "education",
    "github_commits_per_month",
    "patents",
    "open_source_projects",
]


def test_generates_100_pairs():
    pairs = generate_pairs(n=100, seed=42)
    assert len(pairs) == 100


def test_pair_ids_sequential():
    pairs = generate_pairs(n=100, seed=42)
    ids = [p["pair_id"] for p in pairs]
    assert ids == list(range(1, 101))


def test_profiles_identical_within_pairs():
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        a = pair["candidate_a"]
        b = pair["candidate_b"]
        for field in PROFILE_FIELDS:
            assert a[field] == b[field], (
                f"pair_id={pair['pair_id']}: field '{field}' differs: "
                f"{a[field]!r} vs {b[field]!r}"
            )


def test_names_differ_within_pairs():
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        assert pair["candidate_a"]["name"] != pair["candidate_b"]["name"], (
            f"pair_id={pair['pair_id']}: names must differ"
        )


def test_demographic_groups_differ_within_pairs():
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        assert (
            pair["candidate_a"]["demographic_group"]
            != pair["candidate_b"]["demographic_group"]
        )


def test_demographic_groups_are_valid():
    valid = {"white_female", "white_male", "black_female", "black_male"}
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        assert pair["candidate_a"]["demographic_group"] in valid
        assert pair["candidate_b"]["demographic_group"] in valid


def test_groups_gender_matched():
    """White-female must always pair with black-female, and same for male."""
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        ga = pair["candidate_a"]["demographic_group"]
        gb = pair["candidate_b"]["demographic_group"]
        if ga == "white_female":
            assert gb == "black_female"
        elif ga == "white_male":
            assert gb == "black_male"


def test_group_counts_balanced():
    pairs = generate_pairs(n=100, seed=42)
    female = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_female")
    male = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_male")
    assert female == 50
    assert male == 50


def test_reproducible_with_same_seed():
    pairs_a = generate_pairs(n=100, seed=42)
    pairs_b = generate_pairs(n=100, seed=42)
    assert pairs_a == pairs_b


def test_different_seeds_produce_different_output():
    pairs_a = generate_pairs(n=10, seed=42)
    pairs_b = generate_pairs(n=10, seed=99)
    assert pairs_a != pairs_b


def test_verify_identical_profiles_helper():
    pairs = generate_pairs(n=100, seed=42)
    assert _verify_identical_profiles(pairs) is True


def test_skills_is_list():
    pairs = generate_pairs(n=5, seed=42)
    for pair in pairs:
        assert isinstance(pair["candidate_a"]["skills"], list)
        assert isinstance(pair["candidate_b"]["skills"], list)


def test_numeric_fields_in_range():
    pairs = generate_pairs(n=100, seed=42)
    for pair in pairs:
        for cand in (pair["candidate_a"], pair["candidate_b"]):
            assert 1 <= cand["years_experience"] <= 12
            assert 5 <= cand["github_commits_per_month"] <= 60
            assert 0 <= cand["patents"] <= 3
            assert 0 <= cand["open_source_projects"] <= 5
