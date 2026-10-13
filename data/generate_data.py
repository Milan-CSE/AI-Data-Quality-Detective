"""Script to generate a pristine, realistic workforce dataset for benchmarking.
This dataset contains zero intentional errors and serves as the baseline ground-truth.
"""

import os
import numpy as np
import pandas as pd


def generate_clean_workforce_dataset(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    departments = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "HR"]
    dept_weights = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]

    education_levels = ["High School", "Bachelor", "Master", "PhD"]
    edu_weights = [0.10, 0.55, 0.28, 0.07]

    job_levels = ["Entry", "Mid", "Senior", "Lead", "Director"]

    # Generate ages between 22 and 65
    ages = rng.integers(22, 65, size=n_samples)

    # Years experience constrained by age: max experience ~ age - 21
    max_exp = np.maximum(0, ages - 21)
    # Generate experience proportional to max_exp with beta distribution
    exp_ratios = rng.beta(2, 2, size=n_samples)
    years_experience = np.round(max_exp * exp_ratios, 1)

    # Job level loosely correlates with experience
    assigned_levels = []
    for exp in years_experience:
        if exp < 2:
            assigned_levels.append("Entry")
        elif exp < 6:
            assigned_levels.append(rng.choice(["Entry", "Mid"], p=[0.2, 0.8]))
        elif exp < 12:
            assigned_levels.append(rng.choice(["Mid", "Senior"], p=[0.25, 0.75]))
        elif exp < 20:
            assigned_levels.append(rng.choice(["Senior", "Lead"], p=[0.4, 0.6]))
        else:
            assigned_levels.append(rng.choice(["Lead", "Director"], p=[0.5, 0.5]))

    # Base salary by level + experience multiplier + noise
    level_base_salary = {
        "Entry": 55000,
        "Mid": 80000,
        "Senior": 115000,
        "Lead": 145000,
        "Director": 185000,
    }

    dept_multiplier = {
        "Engineering": 1.15,
        "Finance": 1.10,
        "Sales": 1.05,
        "Marketing": 1.00,
        "Operations": 0.95,
        "HR": 0.92,
    }

    dept_choices = rng.choice(departments, size=n_samples, p=dept_weights)
    edu_choices = rng.choice(education_levels, size=n_samples, p=edu_weights)

    salaries = []
    for i in range(n_samples):
        base = level_base_salary[assigned_levels[i]]
        mult = dept_multiplier[dept_choices[i]]
        exp_bonus = years_experience[i] * 1800
        noise = rng.normal(0, 4500)
        salary = base * mult + exp_bonus + noise
        salaries.append(max(38000.0, round(salary, -2)))

    # Performance score: normal distribution around 3.4 (scale 1.0 to 5.0)
    perf_raw = rng.normal(3.4, 0.6, size=n_samples)
    performance_scores = np.clip(np.round(perf_raw, 2), 1.2, 5.0)

    # Credit score: credit score model (e.g. 600 - 840)
    credit_scores = np.clip(
        np.round(rng.normal(715, 45, size=n_samples)).astype(int), 590, 840
    )

    # Remote ratio: 0, 50, or 100
    remote_ratios = rng.choice([0, 50, 100], size=n_samples, p=[0.3, 0.4, 0.3])

    # IDs
    employee_ids = [f"EMP-{10000 + i}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "employee_id": employee_ids,
            "age": ages,
            "years_experience": years_experience,
            "department": dept_choices,
            "education": edu_choices,
            "job_level": assigned_levels,
            "salary": salaries,
            "performance_score": performance_scores,
            "credit_score": credit_scores,
            "remote_ratio": remote_ratios,
        }
    )

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df_clean = generate_clean_workforce_dataset(n_samples=1500, random_seed=42)
    output_path = os.path.join("data", "raw", "workforce_clean.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"Generated clean dataset: {df_clean.shape} -> {output_path}")
