

"""Train and evaluate crop recommendation models.

Usage:
  python train.py --data data/Crop_recommendation.csv

Optional genuine group validation:
  python train.py --data mydata.csv --group-col region
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from model_utils import FEATURES

SCORING = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]


def candidates(seed):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        )
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost unavailable; using Random Forest only.")

    return models


def summarise(cv):
    return {
        metric: round(float(np.mean(cv[f"test_{metric}"])), 4)
        for metric in SCORING
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/Crop_recommendation.csv"
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column containing genuine region or zone IDs",
    )
    parser.add_argument(
        "--out", default="models/crop_model.joblib"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["label"]
    if args.group_col:
        required.append(args.group_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing columns: {missing}. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=required).copy()
    if df.empty:
        raise SystemExit("No valid rows remain after removing missing data.")

    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    print(f"Dataset rows: {len(df)}")
    print(f"Number of crops: {len(encoder.classes_)}")

    # Reserve the test set before model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.seed,
        stratify=y,
    )

    train_df = df.loc[X_train.index]
    results = {}
    model_candidates = candidates(args.seed)

    print("\nTRAINING-SET CROSS-VALIDATION")

    if args.group_col:
        groups = train_df[args.group_col].values
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            raise SystemExit(
                "Grouped validation needs at least two distinct groups."
            )

        cv = GroupKFold(n_splits=min(5, len(unique_groups)))
        cv_name = "leave_region_out"
    else:
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=args.seed
        )
        groups = None
        cv_name = "random_5fold"

    for name, model in model_candidates.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring=SCORING,
            n_jobs=1,
        )

        results[name] = {cv_name: summarise(scores)}
        print(f"\n{name}")
        print(json.dumps(results[name], indent=2))

    # Choose the model using training data only.
    best_name = max(
        results,
        key=lambda name: results[name][cv_name]["f1_macro"],
    )
    print(f"\nSelected model: {best_name}")

    # Fit only on training data before the final test.
    selected_model = model_candidates[best_name]
    selected_model.fit(X_train, y_train)
    predictions = selected_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_macro": round(
            float(report["macro avg"]["precision"]), 4
        ),
        "recall_macro": round(
            float(report["macro avg"]["recall"]), 4
        ),
        "f1_macro": round(
            float(report["macro avg"]["f1-score"]), 4
        ),
    }

    print("\nFINAL HOLD-OUT TEST RESULTS")
    print(json.dumps(test_metrics, indent=2))
    print("\nPer-crop classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    # Refit the selected model on all rows for deployment.
    # The hold-out metrics above remain the evaluation result.
    final_model = candidates(args.seed)[best_name]
    final_model.fit(X, y)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "model": final_model,
        "label_encoder": encoder,
        "features": FEATURES,
        "metrics": results,
        "test_metrics": test_metrics,
        "model_name": best_name,
        "n_rows": len(df),
        "ranges": {
            feature: [
                float(df[feature].min()),
                float(df[feature].max()),
            ]
            for feature in FEATURES
        },
    }

    joblib.dump(bundle, args.out)
    print(f"\nSaved final model to: {args.out}")
    print("Training completed successfully.")

    if not args.group_col:
        print(
            "\nLimitation: random splits do not establish performance "
            "on unseen geographical regions or future seasons."
        )


if __name__ == "__main__":
    main()
