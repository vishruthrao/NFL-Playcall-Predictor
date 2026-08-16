import joblib
from playcall import PlaySituation, predict_situation

model = joblib.load("models/play_call_model.joblib")

# Example: 3rd & 7, opponent 42, Q4 with 6:18 left, offense leads by 3.
situation = PlaySituation(
    down=3,
    ydstogo=7,
    yardline_100=42,
    score_differential=3,
    game_seconds_remaining=378,
    qtr=4,
    posteam="KC",
    defteam="BUF",
    offense_personnel="1 RB, 1 TE, 3 WR",
    posteam_timeouts_remaining=3,
    defteam_timeouts_remaining=3,
    shotgun=1,
    no_huddle=0,
)

result = predict_situation(model, situation)
print(f"Prediction: {result['prediction']}")
print(f"PASS: {result['pass_probability']:.1%}")
print(f"RUN:  {result['run_probability']:.1%}")
