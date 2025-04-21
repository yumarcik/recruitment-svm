import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils import scale_features

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X = df[["experience_years", "technical_score"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler, X_train_scaled, _ = scale_features(X_train, X_test)

    model = SVC(kernel="linear")
    model.fit(X_train_scaled, y_train)

    def plot_decision_boundary(X, y, model):
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.xlabel('Experience (scaled)')
        plt.ylabel('Technical Score (scaled)')
        plt.title('SVM Decision Boundary')
        plt.show()

    plot_decision_boundary(X_train_scaled, y_train, model)