from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target  # 0 = malignant, 1 = benign

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)
log_model.fit(X_train, y_train)

best_depth, best_score = None, -1
for depth in [3, 5, None]:
    candidate = RandomForestClassifier(
        n_estimators=300, max_depth=depth, min_samples_leaf=2,
        random_state=42, class_weight="balanced"
    )
    candidate.fit(X_train, y_train)
    score = accuracy_score(y_val, candidate.predict(X_val))
    if score > best_score:
        best_score, best_depth = score, depth

X_pretest = np.vstack([X_train, X_val])
y_pretest = np.concatenate([y_train, y_val])

log_model.fit(X_pretest, y_pretest)
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=best_depth, min_samples_leaf=2,
    random_state=42, class_weight="balanced"
)
rf_model.fit(X_pretest, y_pretest)
baseline = DummyClassifier(strategy="most_frequent").fit(X_pretest, y_pretest)

for name, model in [
    ("Majority baseline", baseline),
    ("Logistic regression", log_model),
    ("Random forest", rf_model)
]:
    pred = model.predict(X_test)
    print("\\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Malignant precision:", precision_score(y_test, pred, pos_label=0, zero_division=0))
    print("Malignant recall:", recall_score(y_test, pred, pos_label=0, zero_division=0))
    print("Malignant F1:", f1_score(y_test, pred, pos_label=0, zero_division=0))
    print("Confusion matrix [malignant, benign]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))

print("Selected random-forest depth:", best_depth)
