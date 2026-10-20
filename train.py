
import json
import sqlite3
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
import skops.io as skops_io
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_data,
    prepare_xy,
)

# ============================================================
# 1. PROJECT AND MLFLOW CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"
MODEL_DIR = PROJECT_ROOT / "models"
SELECTION_FILE = PROJECT_ROOT / "artifacts" / "model_selection.json"

# Create project directories
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def migrate_legacy_artifact_locations():
    """
    Correct legacy macOS artifact locations in the existing SQLite
    database. Only paths beginning with /Users/ are changed.

    This preserves experiment IDs and run history. Existing artifacts
    are not deleted or moved.
    """
    if not MLFLOW_DB.exists():
        return

    with sqlite3.connect(MLFLOW_DB, timeout=30) as conn:
        tables = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'experiments'
            """
        ).fetchone()

        if not tables:
            return

        experiments = conn.execute(
            """
            SELECT experiment_id, name, artifact_location
            FROM experiments
            """
        ).fetchall()

        for experiment_id, name, artifact_location in experiments:
            if artifact_location and artifact_location.startswith("/Users/"):
                new_location = str(
                    (MLRUNS_DIR / str(experiment_id)).resolve()
                )

                Path(new_location).mkdir(parents=True, exist_ok=True)

                conn.execute(
                    """
                    UPDATE experiments
                    SET artifact_location = ?
                    WHERE experiment_id = ?
                    """,
                    (new_location, experiment_id),
                )

                print(
                    f"Migrated MLflow experiment '{name}' "
                    f"to {new_location}"
                )

        conn.commit()


# Automatically fix legacy paths before MLflow initializes.
# Once migrated, subsequent runs leave the corrected paths unchanged.
migrate_legacy_artifact_locations()

# Use the same SQLite database on every training run.
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

EXPERIMENT_NAME = "heart-disease-classification"


# ============================================================
# 2. MODEL HELPERS
# ============================================================

def _trusted_types(model):
    """Return the untrusted types reported by skops for the model."""
    with tempfile.NamedTemporaryFile(suffix=".skops") as f:
        skops_io.dump(model, f.name)
        return list(skops_io.get_untrusted_types(file=f.name))


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def metrics_for(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def cv_metrics(model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
    )

    return {
        f"cv_{metric}_mean": float(result[f"test_{metric}"].mean())
        for metric in scoring
    }


def log_confusion(model, X_test, y_test, filename):
    """Save a confusion matrix to an absolute project-local path."""
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    path = PLOT_DIR / filename
    plt.savefig(path, dpi=160)
    plt.close()

    return path


# ============================================================
# 3. TRAINING
# ============================================================

def main():
    # Load and prepare data
    df = load_data()
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Reuse the existing experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]),
    }

    results = {}

    # Train and evaluate baseline models
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            test_metrics = metrics_for(model, X_test, y_test)
            cv = cv_metrics(model, X_train, y_train)

            mlflow.log_param("model", name)

            mlflow.log_params({
                key: value
                for key, value in (
                    model.named_steps["model"].get_params().items()
                )
                if isinstance(value, (str, int, float, bool))
            })

            mlflow.log_metrics({**test_metrics, **cv})

            plot = log_confusion(
                model,
                X_test,
                y_test,
                f"{name}_confusion_matrix.png",
            )
            mlflow.log_artifact(str(plot))

            mlflow.sklearn.log_model(
                model,
                "model",
                skops_trusted_types=_trusted_types(model),
            )

            results[name] = {**test_metrics, **cv}
            print(name, results[name])

    # ========================================================
    # 4. RANDOM FOREST HYPERPARAMETER TUNING
    # ========================================================

    base = Pipeline([
        ("preprocess", build_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    grid = GridSearchCV(
        base,
        {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    tuned_metrics = metrics_for(best, X_test, y_test)

    with mlflow.start_run(run_name="random_forest_tuned"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric(
            "cv_best_roc_auc",
            float(grid.best_score_),
        )
        mlflow.log_metrics(tuned_metrics)

        mlflow.sklearn.log_model(
            best,
            "model",
            skops_trusted_types=_trusted_types(best),
        )

        plot = log_confusion(
            best,
            X_test,
            y_test,
            "random_forest_tuned_confusion_matrix.png",
        )
        mlflow.log_artifact(str(plot))

    print("Best Random Forest parameters:", grid.best_params_)
    print("Tuned Random Forest test metrics:", tuned_metrics)

    # ========================================================
    # 5. SELECT AND SAVE THE PRODUCTION MODEL
    # ========================================================

    # Select the baseline model using cross-validation ROC-AUC.
    # Keep the existing model-selection strategy.
    cv_roc = {
        "logistic_regression": results[
            "logistic_regression"
        ]["cv_roc_auc_mean"],
        "random_forest": results[
            "random_forest"
        ]["cv_roc_auc_mean"],
    }

    selected_name = max(cv_roc, key=cv_roc.get)
    selected_model = models[selected_name]

    # Refit the selected baseline pipeline on all available data.
    selected_model.fit(X, y)

    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(selected_model, model_path)

    selection_data = {
        "selected_model": selected_name,
        "cv_roc_auc": cv_roc,
        "tuned_random_forest_test_metrics": tuned_metrics,
    }

    with SELECTION_FILE.open("w") as f:
        json.dump(selection_data, f, indent=2)

    print(f"Saved production pipeline: {selected_name}")
    print(f"Model path: {model_path}")
    print(f"Model selection file: {SELECTION_FILE}")
    print(f"MLflow tracking database: {MLFLOW_DB}")


if __name__ == "__main__":
    main()
