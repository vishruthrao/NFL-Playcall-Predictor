# NFL Play Call Predictor

A complete machine-learning project that predicts whether an NFL offense will call a **RUN** or **PASS** from the situation before the snap.

## What the model knows

The initial model uses only information available before the play:

- down
- yards to go
- field position (`yardline_100`)
- score differential
- game time remaining
- quarter
- offense and defense
- offensive personnel
- offense/defense timeouts
- shotgun
- no-huddle

The target is `play_type`: PASS = 1, RUN = 0. QB kneels and spikes are removed.

## Why the train/test split matters

The split is chronological, not random. For example, train on 2021-2023 and test on 2024. This prevents future plays from leaking into the training sample and gives a more realistic estimate of performance on unseen football.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
```

## Train

Easy mode:

```bash
python train.py
```

Or select seasons yourself:

```bash
python playcall.py train --seasons 2021 2022 2023 2024 --test-season 2024
```

`nflreadpy` downloads the nflverse play-by-play data automatically.

Artifacts created:

- `models/play_call_model.joblib`
- `outputs/metrics.json`

## Make a prediction

Edit `predict.py`, then:

```bash
python predict.py
```

Or use the CLI directly:

```bash
python playcall.py predict \
  --down 3 \
  --ydstogo 7 \
  --yardline-100 42 \
  --score-differential 3 \
  --game-seconds-remaining 378 \
  --qtr 4 \
  --posteam KC \
  --defteam BUF \
  --offense-personnel "1 RB, 1 TE, 3 WR" \
  --shotgun 1
```

Output looks like:

```json
{
  "prediction": "PASS",
  "pass_probability": 0.73,
  "run_probability": 0.27
}
```

The actual numbers depend on the trained NFL data.

## Feature importance

```bash
python plot_importance.py
```

This writes `outputs/feature_importance.png`.

## Test the code without downloading NFL data

```bash
pytest -q
```

The unit test creates synthetic play-by-play data and verifies the full preprocessing, training, saving, and prediction path.

## Next useful upgrades

1. Team-specific models or team/coach rolling tendencies.
2. Formation and receiver alignment features where available.
3. Previous-play and drive-context features that are known before the next snap.
4. Probability calibration.
5. Walk-forward evaluation by week instead of only by season.
6. A Streamlit web interface with dropdowns and sliders.
7. Multiclass prediction for inside run / outside run / screen / short pass / intermediate pass / deep pass, subject to label quality in the source data.

## Web app

A Streamlit website is included as `app.py`.

Run it locally:

```bash
streamlit run app.py
```

See `DEPLOY.md` for the easiest public deployment route.
