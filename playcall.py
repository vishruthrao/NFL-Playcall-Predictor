from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "shotgun",
    "no_huddle",
]

CATEGORICAL_FEATURES = [
    "posteam",
    "defteam",
    "offense_personnel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_pass"


@dataclass
class PlaySituation:
    down: int
    ydstogo: float
    yardline_100: float
    score_differential: float
    game_seconds_remaining: float
    qtr: int
    posteam: str = "UNK"
    defteam: str = "UNK"
    offense_personnel: str = "UNK"
    posteam_timeouts_remaining: int = 3
    defteam_timeouts_remaining: int = 3
    shotgun: int = 0
    no_huddle: int = 0


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Download nflverse play-by-play using nflreadpy."""
    import nflreadpy as nfl

    frame = nfl.load_pbp(list(seasons))
    return frame.to_pandas()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional columns when a season/version omits them."""
    df = df.copy()
    defaults = {
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "shotgun": 0,
        "no_huddle": 0,
        "offense_personnel": "UNK",
        "posteam": "UNK",
        "defteam": "UNK",
        "qb_kneel": 0,
        "qb_spike": 0,
        "season": np.nan,
        "week": np.nan,
        "game_id": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def prepare_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep normal offensive run/pass plays and construct only pre-snap features.

    PASS is encoded as 1 and RUN as 0.
    Kneels and spikes are excluded because they are strategic end-game actions
    rather than ordinary play calls.
    """
    df = _ensure_columns(df)

    if "play_type" not in df.columns:
        raise ValueError("Expected nflverse column 'play_type' was not found.")

    plays = df[df["play_type"].isin(["run", "pass"])].copy()
    plays = plays[plays["qb_kneel"].fillna(0).eq(0)]
    plays = plays[plays["qb_spike"].fillna(0).eq(0)]

    plays[TARGET] = plays["play_type"].eq("pass").astype(int)

    for col in NUMERIC_FEATURES:
        plays[col] = pd.to_numeric(plays[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        plays[col] = plays[col].fillna("UNK").astype(str)

    required = ["down", "ydstogo", "yardline_100", "game_seconds_remaining"]
    plays = plays.dropna(subset=required + [TARGET])

    numeric_fill = {
        "score_differential": 0,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "shotgun": 0,
        "no_huddle": 0,
        "qtr": 1,
    }
    for col, value in numeric_fill.items():
        plays[col] = plays[col].fillna(value)

    plays["down"] = plays["down"].clip(1, 4)
    plays["qtr"] = plays["qtr"].clip(lower=1)
    plays["shotgun"] = plays["shotgun"].astype(int)
    plays["no_huddle"] = plays["no_huddle"].astype(int)

    return plays


def chronological_split(
    plays: pd.DataFrame, test_season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(int(x) for x in plays["season"].dropna().unique())
    if not seasons:
        raise ValueError("No season values are available for chronological splitting.")

    test_season = test_season or seasons[-1]
    train = plays[plays["season"] < test_season].copy()
    test = plays[plays["season"] == test_season].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Need at least one training season before test season {test_season}. "
            f"Available seasons: {seasons}"
        )
    return train, test


def build_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=550,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.5,
        reg_alpha=0.05,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline([("prep", preprocessor), ("model", model)])


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict:
    X_test = test[FEATURES]
    y_test = test[TARGET]
    p_pass = model.predict_proba(X_test)[:, 1]
    pred = (p_pass >= 0.5).astype(int)

    metrics = {
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, p_pass)),
        "log_loss": float(log_loss(y_test, p_pass)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=["RUN", "PASS"], output_dict=True
        ),
    }
    return metrics


def train_from_dataframe(
    df: pd.DataFrame,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    test_season: int | None = None,
) -> tuple[Pipeline, dict]:
    plays = prepare_plays(df)
    train, test = chronological_split(plays, test_season=test_season)

    model = build_pipeline()
    model.fit(train[FEATURES], train[TARGET])
    metrics = evaluate(model, test)
    metrics.update(
        {
            "train_seasons": sorted(int(x) for x in train["season"].unique()),
            "test_season": int(test["season"].iloc[0]),
            "n_train": int(len(train)),
            "pass_rate_train": float(train[TARGET].mean()),
            "pass_rate_test": float(test[TARGET].mean()),
        }
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))

    return model, metrics


def train_from_nflverse(
    seasons: Iterable[int],
    model_path: str | Path = "models/play_call_model.joblib",
    metrics_path: str | Path = "outputs/metrics.json",
    test_season: int | None = None,
) -> tuple[Pipeline, dict]:
    pbp = load_pbp(seasons)
    return train_from_dataframe(pbp, model_path, metrics_path, test_season)


def predict_situation(model: Pipeline, situation: PlaySituation) -> dict:
    row = pd.DataFrame([asdict(situation)])
    p_pass = float(model.predict_proba(row[FEATURES])[:, 1][0])
    p_run = 1.0 - p_pass
    return {
        "prediction": "PASS" if p_pass >= 0.5 else "RUN",
        "pass_probability": p_pass,
        "run_probability": p_run,
    }


def feature_importance(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    booster = model.named_steps["model"]
    names = prep.get_feature_names_out()
    values = booster.feature_importances_
    result = pd.DataFrame({"feature": names, "importance": values})
    return result.sort_values("importance", ascending=False).head(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL RUN/PASS play-call predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Download nflverse data and train the model")
    train_p.add_argument("--seasons", nargs="+", type=int, required=True)
    train_p.add_argument("--test-season", type=int, default=None)
    train_p.add_argument("--model", default="models/play_call_model.joblib")
    train_p.add_argument("--metrics", default="outputs/metrics.json")

    pred_p = sub.add_parser("predict", help="Predict RUN vs PASS for a situation")
    pred_p.add_argument("--model", default="models/play_call_model.joblib")
    pred_p.add_argument("--down", type=int, required=True)
    pred_p.add_argument("--ydstogo", type=float, required=True)
    pred_p.add_argument("--yardline-100", type=float, required=True)
    pred_p.add_argument("--score-differential", type=float, required=True)
    pred_p.add_argument("--game-seconds-remaining", type=float, required=True)
    pred_p.add_argument("--qtr", type=int, required=True)
    pred_p.add_argument("--posteam", default="UNK")
    pred_p.add_argument("--defteam", default="UNK")
    pred_p.add_argument("--offense-personnel", default="UNK")
    pred_p.add_argument("--posteam-timeouts", type=int, default=3)
    pred_p.add_argument("--defteam-timeouts", type=int, default=3)
    pred_p.add_argument("--shotgun", type=int, choices=[0, 1], default=0)
    pred_p.add_argument("--no-huddle", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    if args.command == "train":
        _, metrics = train_from_nflverse(
            seasons=args.seasons,
            model_path=args.model,
            metrics_path=args.metrics,
            test_season=args.test_season,
        )
        print(json.dumps(metrics, indent=2))
    else:
        model = joblib.load(args.model)
        situation = PlaySituation(
            down=args.down,
            ydstogo=args.ydstogo,
            yardline_100=args.yardline_100,
            score_differential=args.score_differential,
            game_seconds_remaining=args.game_seconds_remaining,
            qtr=args.qtr,
            posteam=args.posteam,
            defteam=args.defteam,
            offense_personnel=args.offense_personnel,
            posteam_timeouts_remaining=args.posteam_timeouts,
            defteam_timeouts_remaining=args.defteam_timeouts,
            shotgun=args.shotgun,
            no_huddle=args.no_huddle,
        )
        print(json.dumps(predict_situation(model, situation), indent=2))


if __name__ == "__main__":
    main()
