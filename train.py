from playcall import train_from_nflverse, feature_importance

# Train on 2021-2024 and hold 2024 out as the true future test season.
model, metrics = train_from_nflverse(
    seasons=[2021, 2022, 2023, 2024],
    test_season=2024,
    model_path="models/play_call_model.joblib",
    metrics_path="outputs/metrics.json",
)

print("\nModel evaluation")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC AUC:  {metrics['roc_auc']:.3f}")
print(f"Log loss: {metrics['log_loss']:.3f}")
print("\nTop features")
print(feature_importance(model, top_n=20).to_string(index=False))
