from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from playcall import PlaySituation, feature_importance, predict_situation, train_from_nflverse

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "play_call_model.joblib"
METRICS_PATH = APP_DIR / "outputs" / "metrics.json"

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

PERSONNEL = [
    "1 RB, 1 TE, 3 WR",  # 11 personnel
    "1 RB, 2 TE, 2 WR",  # 12
    "2 RB, 1 TE, 2 WR",  # 21
    "2 RB, 2 TE, 1 WR",  # 22
    "1 RB, 3 TE, 1 WR",  # 13
    "0 RB, 1 TE, 4 WR",  # 01
    "0 RB, 0 TE, 5 WR",  # 00
]

st.set_page_config(
    page_title="NFL Play Call Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1180px;}
        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(128,128,128,.28);
            border-radius: 18px;
            margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0 0 .25rem 0; font-size: 2.25rem;}
        .hero p {margin: 0; opacity: .78; font-size: 1.02rem;}
        .prediction-card {
            border: 1px solid rgba(128,128,128,.28);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            margin-top: .4rem;
        }
        .prediction-label {font-size: .85rem; opacity: .68; text-transform: uppercase; letter-spacing: .08em;}
        .prediction-main {font-size: 3.1rem; font-weight: 800; line-height: 1.05; margin: .2rem 0 .8rem 0;}
        .small-note {font-size: .88rem; opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_saved_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


if "model" not in st.session_state:
    st.session_state.model = load_saved_model()

st.markdown(
    """
    <div class="hero">
        <h1>🏈 NFL Play Call Predictor</h1>
        <p>Use the pre-snap game situation to estimate whether the offense will call a RUN or PASS.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = st.session_state.model

if model is None:
    st.warning("No trained model is saved yet. Train one below, then the predictor will unlock.")
    with st.expander("Train the model", expanded=True):
        st.write(
            "This downloads nflverse play-by-play data, trains on earlier seasons, "
            "and tests on the latest season you select."
        )
        seasons = st.multiselect(
            "NFL seasons",
            options=list(range(2018, 2027)),
            default=[2021, 2022, 2023, 2024],
        )
        if len(seasons) >= 2:
            test_season = st.selectbox("Hold-out test season", options=sorted(seasons), index=len(seasons) - 1)
        else:
            test_season = None
        train_clicked = st.button("Train model", type="primary", use_container_width=True)
        if train_clicked:
            if len(seasons) < 2:
                st.error("Choose at least two seasons so one can be held out for testing.")
            elif min(seasons) == test_season:
                st.error("The test season needs at least one earlier season for training.")
            else:
                with st.spinner("Downloading NFL plays and training the model..."):
                    try:
                        model, metrics = train_from_nflverse(
                            seasons=sorted(seasons),
                            test_season=int(test_season),
                            model_path=MODEL_PATH,
                            metrics_path=METRICS_PATH,
                        )
                        st.session_state.model = model
                        st.success(
                            f"Model trained. Test accuracy: {metrics['accuracy']:.1%} | "
                            f"ROC AUC: {metrics['roc_auc']:.3f}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Game situation")
    offense = st.selectbox("Offense", TEAMS, index=TEAMS.index("KC"))
    defense_options = [t for t in TEAMS if t != offense]
    defense = st.selectbox("Defense", defense_options, index=defense_options.index("BUF") if "BUF" in defense_options else 0)

    st.divider()
    down = st.select_slider("Down", options=[1, 2, 3, 4], value=3)
    ydstogo = st.slider("Yards to first down", min_value=1, max_value=30, value=7)
    yardline_100 = st.slider(
        "Yards from opponent end zone",
        min_value=1,
        max_value=99,
        value=42,
        help="NFL play-by-play yardline_100. 1 means almost at the goal line; 99 means backed up near your own goal line.",
    )

    st.divider()
    qtr = st.selectbox("Quarter", [1, 2, 3, 4, 5], index=3, format_func=lambda x: "OT" if x == 5 else f"Q{x}")
    c1, c2 = st.columns(2)
    with c1:
        minutes = st.number_input("Minutes", min_value=0, max_value=15, value=6, step=1)
    with c2:
        seconds = st.number_input("Seconds", min_value=0, max_value=59, value=18, step=1)

    if qtr <= 4:
        game_seconds_remaining = (4 - qtr) * 900 + int(minutes) * 60 + int(seconds)
    else:
        game_seconds_remaining = int(minutes) * 60 + int(seconds)

    score_diff = st.slider(
        "Offense score differential",
        min_value=-35,
        max_value=35,
        value=3,
        help="Positive means the offense is leading; negative means it is trailing.",
    )

    st.divider()
    personnel = st.selectbox("Offensive personnel", PERSONNEL, index=0)
    shotgun = st.toggle("Shotgun", value=True)
    no_huddle = st.toggle("No huddle", value=False)
    offense_timeouts = st.select_slider("Offense timeouts", options=[0, 1, 2, 3], value=3)
    defense_timeouts = st.select_slider("Defense timeouts", options=[0, 1, 2, 3], value=3)

situation = PlaySituation(
    down=int(down),
    ydstogo=float(ydstogo),
    yardline_100=float(yardline_100),
    score_differential=float(score_diff),
    game_seconds_remaining=float(game_seconds_remaining),
    qtr=int(qtr),
    posteam=offense,
    defteam=defense,
    offense_personnel=personnel,
    posteam_timeouts_remaining=int(offense_timeouts),
    defteam_timeouts_remaining=int(defense_timeouts),
    shotgun=int(shotgun),
    no_huddle=int(no_huddle),
)

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.subheader("Current situation")
    situation_cols = st.columns(4)
    situation_cols[0].metric("Down", f"{down}")
    situation_cols[1].metric("Distance", f"{ydstogo} yd")
    situation_cols[2].metric("Field position", f"{yardline_100} yd out")
    situation_cols[3].metric("Score diff", f"{score_diff:+d}")

    situation_cols2 = st.columns(4)
    situation_cols2[0].metric("Quarter", "OT" if qtr == 5 else f"Q{qtr}")
    situation_cols2[1].metric("Clock", f"{int(minutes):02d}:{int(seconds):02d}")
    situation_cols2[2].metric("Offense", offense)
    situation_cols2[3].metric("Defense", defense)

    st.caption(f"Personnel: {personnel} · {'Shotgun' if shotgun else 'Under center'} · {'No huddle' if no_huddle else 'Huddle'}")

    predict_clicked = st.button("🏈 Predict Play", type="primary", use_container_width=True)

    if predict_clicked or "last_prediction" in st.session_state:
        if predict_clicked:
            st.session_state.last_prediction = predict_situation(model, situation)
        result = st.session_state.last_prediction
        p_pass = result["pass_probability"]
        p_run = result["run_probability"]

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted play call</div>
                <div class="prediction-main">{result['prediction']}</div>
                <div class="small-note">Model probability, not certainty. Football decisions are contextual and can change with personnel, coaching, injuries, and game plan.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        cpass, crun = st.columns(2)
        cpass.metric("PASS", f"{p_pass:.1%}")
        crun.metric("RUN", f"{p_run:.1%}")
        st.write("**PASS probability**")
        st.progress(float(p_pass))
        st.write("**RUN probability**")
        st.progress(float(p_run))
    else:
        st.info("Adjust the game situation, then click **Predict Play**.")

with right:
    st.subheader("Model")
    metrics = load_metrics()
    if metrics:
        m1, m2 = st.columns(2)
        m1.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.caption(
            f"Trained on {', '.join(map(str, metrics.get('train_seasons', [])))} · "
            f"tested on {metrics.get('test_season', 'N/A')} · "
            f"{metrics.get('n_train', 0):,} training plays"
        )
    else:
        st.caption("Saved model loaded. Evaluation metrics are not available in this deployment.")

    st.write("**What the model considers**")
    st.write(
        "Down, distance, field position, score differential, time remaining, quarter, "
        "offense, defense, personnel, timeouts, shotgun, and no-huddle."
    )

    with st.expander("Top learned features"):
        try:
            fi = feature_importance(model, top_n=12).copy()
            fi["feature"] = (
                fi["feature"]
                .str.replace("cat__", "", regex=False)
                .str.replace("num__", "", regex=False)
            )
            st.dataframe(fi, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.caption(f"Feature importance unavailable: {exc}")

    with st.expander("How to read this"):
        st.write(
            "A 72% PASS result means the model believes similar historical pre-snap situations "
            "were more consistent with a pass than a run. It does not mean the offense has a "
            "72% chance of completing a pass or winning the play."
        )

st.divider()
st.caption(
    "Built for educational analysis using nflverse play-by-play data. Predictions are estimates from historical patterns, not inside information or betting advice."
)
