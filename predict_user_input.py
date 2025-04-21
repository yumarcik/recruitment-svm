import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils import scale_features

def is_valid_input(value: float, min_val: float, max_val: float) -> bool:
    return isinstance(value, (int, float)) and min_val <= value <= max_val

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df[["experience_years", "technical_score"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler, X_train_scaled, _ = scale_features(X_train, X_test)

    model = SVC(kernel="linear")
    model.fit(X_train_scaled, y_train)

    try:
        exp = float(input("Experience (years, 0.0 - 10.0): "))
        score = float(input("Technical Score (0.0 - 100.0): "))

        if not is_valid_input(exp, 0.0, 10.0):
            raise ValueError("Experience must be between 0.0 and 10.0")

        if not is_valid_input(score, 0.0, 100.0):
            raise ValueError("Technical score must be between 0.0 and 100.0")

        input_df = pd.DataFrame([[exp, score]], columns=["experience_years", "technical_score"])
        user_scaled = scaler.transform(input_df)

        result = model.predict(user_scaled)[0]
        print("Prediction:", "Accepted" if result == 0 else "Rejected")

    except ValueError as e:
        print(e)
    except:
        print("Please enter valid numeric values.")
