import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

TARGET = "Potability"


def prepare_dataset(df):

    df = df.copy()

    df = df.drop_duplicates()

    X = df[FEATURES]

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        scaler
    )


def get_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                metric="euclidean"
            )
    }

    return models


def train_models(df):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        imputer,
        scaler
    ) = prepare_dataset(df)

    models = get_models()

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        results[name] = {

            "model": model,

            "accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "roc_auc":
                roc_auc_score(
                    y_test,
                    y_probability
                ),

            "y_test": y_test,

            "y_pred": y_pred,

            "y_probability":
                y_probability,

            "imputer": imputer,

            "scaler": scaler
        }

    return results