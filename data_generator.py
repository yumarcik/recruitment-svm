import pandas as pd
import numpy as np
from faker import Faker

def generate_data(n=200, seed=42):
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    experience_choices = np.arange(0, 10.5 + 0.5, 0.5)
    score_choices = np.arange(0, 100.5 + 0.5, 0.5)

    records = []

    for _ in range(n):
        name = fake.name()
        experience = float(np.random.choice(experience_choices))
        technical_score = float(np.random.choice(score_choices))

        label = 1 if experience < 2.0 and technical_score < 60.0 else 0

        records.append({
            "name": name,
            "experience_years": experience,
            "technical_score": technical_score,
            "label": label
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data.csv", index=False)
    print("Data generated and saved to 'data.csv'.")