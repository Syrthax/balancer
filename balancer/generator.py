import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
NAMES_PATH = REPO_ROOT / "data" / "names.json"
OUTPUT_PATH = REPO_ROOT / "data" / "candidates.json"

SKILLS_POOL = [
    "Python", "React", "ML", "Docker", "Kubernetes",
    "Go", "Rust", "SQL", "TensorFlow", "FastAPI",
    "TypeScript", "Java", "C++", "Redis", "PostgreSQL",
    "AWS", "GCP", "Spark", "PyTorch", "Node.js",
]

EDUCATIONS = [
    "BSc Computer Science",
    "MSc Computer Science",
    "BSc Software Engineering",
    "MSc Data Science",
    "BSc Mathematics",
    "MSc Machine Learning",
    "BSc Electrical Engineering",
    "BSc Information Technology",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Jones", "Brown",
    "Davis", "Miller", "Wilson", "Moore", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
    "Clark", "Rodriguez", "Lewis", "Lee", "Walker",
]


def _load_names() -> dict[str, list[str]]:
    with open(NAMES_PATH) as f:
        return json.load(f)


def generate_pairs(n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = _load_names()

    pairs: list[dict[str, Any]] = []

    # Split evenly: 50 female pairs, 50 male pairs → 25 WF/BF + 25 WM/BM per gender
    female_count = n // 2
    male_count = n - female_count

    combos = (
        [("white_female", "black_female")] * female_count
        + [("white_male", "black_male")] * male_count
    )
    rng.shuffle(combos)

    for pair_id, (group_a, group_b) in enumerate(combos, start=1):
        # Generate shared profile values
        years_exp = rng.randint(1, 12)
        num_skills = rng.randint(2, 5)
        skills = sorted(rng.sample(SKILLS_POOL, num_skills))
        education = rng.choice(EDUCATIONS)
        commits = rng.randint(5, 60)
        patents = rng.randint(0, 3)
        oss = rng.randint(0, 5)

        # Pick names independently so they don't repeat within the pair
        name_a = rng.choice(names[group_a])
        name_b = rng.choice(names[group_b])
        last_name = rng.choice(LAST_NAMES)

        pairs.append({
            "pair_id": pair_id,
            "candidate_a": {
                "name": f"{name_a} {last_name}",
                "demographic_group": group_a,
                "years_experience": years_exp,
                "skills": skills,
                "education": education,
                "github_commits_per_month": commits,
                "patents": patents,
                "open_source_projects": oss,
            },
            "candidate_b": {
                "name": f"{name_b} {last_name}",
                "demographic_group": group_b,
                "years_experience": years_exp,
                "skills": skills,
                "education": education,
                "github_commits_per_month": commits,
                "patents": patents,
                "open_source_projects": oss,
            },
        })

    return pairs


def save_pairs(pairs: list[dict[str, Any]], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)


def load_pairs(path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _verify_identical_profiles(pairs: list[dict[str, Any]]) -> bool:
    profile_fields = [
        "years_experience", "skills", "education",
        "github_commits_per_month", "patents", "open_source_projects",
    ]
    for pair in pairs:
        a, b = pair["candidate_a"], pair["candidate_b"]
        for field in profile_fields:
            if a[field] != b[field]:
                return False
    return True
