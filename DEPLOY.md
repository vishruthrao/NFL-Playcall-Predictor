# Put the NFL Play Call Predictor on the web

## Easiest route: Streamlit Community Cloud

1. Create a GitHub repository, for example `nfl-play-call-predictor`.
2. Upload every file from this project to the repository.
3. Train the model once by running `python train.py` on your computer or in a browser environment that can run Python.
4. Make sure these generated files are also in the repository:
   - `models/play_call_model.joblib`
   - `outputs/metrics.json`
5. Sign in to Streamlit Community Cloud with GitHub.
6. Create an app and select your repository.
7. Use `app.py` as the entrypoint file.
8. Deploy.

Your site will receive a `*.streamlit.app` URL that you can share.

## Run it on your own computer

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The browser will open the app, normally at `http://localhost:8501`.

## Important deployment note

The app includes an in-app training screen if a saved model is not present. For a public site, however, the better setup is to train once and deploy the saved model artifact so visitors can predict immediately and the model does not need to retrain after a server restart.
