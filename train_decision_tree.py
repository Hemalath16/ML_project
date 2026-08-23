from preprocessing import prepare_data

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib


X_train, X_test, y_train, y_test, imputer, scaler = prepare_data()

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("=" * 50)
print("DECISION TREE RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print()
print("Classification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "scaler": scaler
    },
    "models/decision_tree.pkl"
)

print()
print("Decision Tree model saved successfully!")