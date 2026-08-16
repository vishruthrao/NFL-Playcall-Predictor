import numpy as np
import pandas as pd

from playcall import PlaySituation, predict_situation, prepare_plays, train_from_dataframe


def synthetic_pbp(n=800, seed=7):
    rng = np.random.default_rng(seed)
    season = np.where(np.arange(n) < int(n * 0.75), 2023, 2024)
    down = rng.integers(1, 5, n)
    ydstogo = rng.integers(1, 16, n)
    yardline = rng.integers(1, 100, n)
    score = rng.integers(-21, 22, n)
    seconds = rng.integers(1, 3601, n)
    shotgun = rng.integers(0, 2, n)
    logits = -1.5 + 0.55 * (down >= 3) + 0.09 * ydstogo + 0.8 * shotgun - 0.025 * score
    p = 1 / (1 + np.exp(-logits))
    is_pass = rng.random(n) < p
    return pd.DataFrame({
        "season": season,
        "week": rng.integers(1, 19, n),
        "game_id": [f"g{i//60}" for i in range(n)],
        "play_type": np.where(is_pass, "pass", "run"),
        "qb_kneel": 0,
        "qb_spike": 0,
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline,
        "score_differential": score,
        "game_seconds_remaining": seconds,
        "qtr": np.minimum(4, ((3600-seconds)//900)+1),
        "posteam": rng.choice(["KC", "BUF", "PHI", "SF"], n),
        "defteam": rng.choice(["DAL", "BAL", "DET", "MIA"], n),
        "offense_personnel": rng.choice(["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR"], n),
        "posteam_timeouts_remaining": rng.integers(0, 4, n),
        "defteam_timeouts_remaining": rng.integers(0, 4, n),
        "shotgun": shotgun,
        "no_huddle": rng.integers(0, 2, n),
    })


def test_prepare_and_train(tmp_path):
    df = synthetic_pbp()
    plays = prepare_plays(df)
    assert set(plays["is_pass"].unique()).issubset({0, 1})

    model_path = tmp_path / "model.joblib"
    model, metrics = train_from_dataframe(df, model_path, test_season=2024)
    assert model_path.exists()
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1

    result = predict_situation(model, PlaySituation(
        down=3, ydstogo=8, yardline_100=45, score_differential=-3,
        game_seconds_remaining=420, qtr=4, posteam="KC", defteam="BUF",
        offense_personnel="1 RB, 1 TE, 3 WR", shotgun=1,
    ))
    assert result["prediction"] in {"RUN", "PASS"}
    assert abs(result["pass_probability"] + result["run_probability"] - 1) < 1e-9
