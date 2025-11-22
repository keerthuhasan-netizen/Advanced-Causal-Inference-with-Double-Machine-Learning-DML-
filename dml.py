
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def generate_data(n=1500, p=20, seed=42):
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, p))

    # Treatment assignment (depends on X)
    T = (X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 1, n) > 0).astype(int)

    # Outcome (depends on X and treatment effect)
    tau = 3 + X[:, 2] - 0.7 * X[:, 3]
    Y = tau * T + X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 1, n)

    return pd.DataFrame(np.column_stack([Y, T, X]), columns=["Y", "T"] + [f"X{i}" for i in range(p)])

def dml(X, T, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_residuals = np.zeros(len(Y))
    t_residuals = np.zeros(len(T))

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        # Outcome model
        y_model = RandomForestRegressor(n_estimators=200)
        y_model.fit(X_train, Y_train)
        y_pred = y_model.predict(X_test)

        # Treatment model
        t_model = RandomForestRegressor(n_estimators=200)
        t_model.fit(X_train, T_train)
        t_pred = t_model.predict(X_test)

        y_residuals[test_idx] = Y_test - y_pred
        t_residuals[test_idx] = T_test - t_pred

    # Final stage regression
    final_model = LinearRegression()
    final_model.fit(t_residuals.reshape(-1, 1), y_residuals)
    ate = final_model.coef_[0]
    se = np.std(y_residuals - final_model.predict(t_residuals.reshape(-1, 1))) / np.sqrt(len(Y))

    return ate, se

def main():
    df = generate_data()
    Y = df["Y"].values
    T = df["T"].values
    X = df[[c for c in df.columns if c.startswith("X")]].values

    ate, se = dml(X, T, Y)
    print("\nEstimated ATE:", ate)
    print("Standard Error:", se)

if __name__ == "__main__":
    main()
