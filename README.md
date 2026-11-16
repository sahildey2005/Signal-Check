# Signal Check

A local browser prototype that compares a headline with two separate models: clickbait vs. non-clickbait, and fake-labeled vs. true-labeled examples.

## Run it

On Windows, double-click `run.bat`. It starts a local web server and opens the app. Keep the command window open while using it; press Ctrl+C there to stop the server. Python is required. Open the app through `run.bat`, not by double-clicking `index.html`, so the browser can load `models.json`.

## Models and datasets

- `clickbait_data.csv` supplies the `headline` and `clickbait` columns. The model reads `1` as clickbait and `0` as non-clickbait.
- `Fake.csv` and `True.csv` supply the `title` column. The filename supplies the training label. The model only sees headlines, even though article text is present in the files.
- `train_models.py` deduplicates normalized headlines, creates a 70% training / 15% calibration / 15% holdout split, trains two multinomial Naive Bayes models, and writes `models.json`.
- The training script uses Python's standard library only. It defaults to `data/clickbait_data.csv`, `data/Fake.csv`, and `data/True.csv` next to the script; pass `--clickbait`, `--fake`, `--true`, or `--output` to use different paths.
- The generated model file contains token weights and reports, not copies of the source article rows.

To retrain, place your CSV files in a local `data` folder (which is ignored by Git) and run:

```text
python train_models.py
```

## Current holdout results

On the held-out part of the supplied files, the clickbait model reached **97.8% accuracy** over 4,800 headlines; the fake/true model reached **95.5% accuracy** over 5,809 headlines. These results measure fit to the supplied datasets and do not predict accuracy on current news or other sources.

## Limits

The fake/true labels are inherited from the filenames and are not independent fact-checks. Both models can learn quirks of the datasets, such as topic, time period, and publisher style. Their percentages are model estimates, not calibrated real-world probabilities. A headline alone cannot establish that a story is true or false. Use the related-coverage and fact-check searches to compare reporting and evidence.

Headlines are analyzed locally in the browser. When a user deliberately submits labelled feedback, the headline, chosen labels, prediction scores, and (for fake/true labels) evidence URL are sent to the local feedback server and stored in `feedback.sqlite3`. Search terms are sent to a search provider only if a coverage link is opened. The included server is for local development and binds to 127.0.0.1; GitHub Pages cannot run this backend. A separately hosted API must be deployed and configured in `config.js` before remote visitors can submit feedback.


## Developer checks

Run the dependency-free regression tests before changing the trainer:

```text
python -m py_compile train_models.py
python -m unittest discover -s tests -v
```

The test workflow also runs these checks on Python 3.10 and 3.12. The trainer reports holdout accuracy, balanced accuracy, per-class precision/recall/F1, macro F1, specificity, Brier score, and log loss. These metrics still describe only the supplied holdout split; they are not evidence of real-world fact-checking accuracy.

The optimized trainer keeps the existing browser model fields. It makes two feature-extraction passes during training to avoid storing every headline's sparse features in memory, reuses calibration/holdout scores, and stops calibration early when its gradients converge.

## Interface features

- Two separate headline checks: clickbait-style wording and fake-/true-labelled dataset similarity.
- Evidence cues show influential words, phrases, and headline-style features.
- Holdout evaluation metrics and confusion matrices are displayed for both models.
- Copy results lets users copy a short summary of the current analysis.
- A labelled-feedback panel lets users suggest corrected labels for the clickbait and fake/true dataset-pattern models. Feedback is stored in a local SQLite database and remains pending until a project owner reviews it.
- `manage_feedback.py` lists, approves, and rejects submissions. Only manually approved labels can be included in a later training run.
- After at least 10 approved labels exist for each classifier, approving a further submission automatically builds `models_candidate.json` (if the three original CSVs are present). The current `models.json` is never overwritten automatically.
- You can also manually run `train_models.py --feedback-db feedback.sqlite3 --output models_candidate.json` to retrain both Naive Bayes models using the original CSVs plus approved feedback.
- A separate, cautious browser-local self-learning memory stores feature patterns from very high-confidence clickbait predictions (92% or higher). Once at least three similar remembered examples exist, those patterns make a small adjustment to later clickbait scores. It stores feature tokens rather than original headlines, keeps at most 200 examples, and does not upload data or alter the original `models.json` model.
- The older browser-local clickbait memory is still experimental self-training, not verified learning: it uses its own high-confidence predictions as provisional labels and can reinforce mistakes. It is separate from the new supervised feedback queue.
- The project guide explains the difference between clickbait and misinformation, the training pipeline, technologies, and limitations.
- Related coverage and fact-check search links help users continue checking a claim.

The confusion matrices and evaluation metrics are read from the reports generated by `train_models.py`. They describe the supplied holdout datasets only. The browser interface does not perform independent fact-checking.


## Supervised feedback and retraining

The feedback system is deliberately review-first. A user can choose a corrected clickbait label and/or a fake/true dataset-pattern label. A source URL is required when a fake/true label is submitted. The server stores submissions as `pending`; user submissions are not trusted training data by default.

1. Run the local app with `run.bat` and collect feedback.
2. In the project folder, run `python manage_feedback.py list` to inspect pending submissions and evidence.
3. After independently checking a submission, approve it with `python manage_feedback.py approve ID --confirm-reviewed`. Reject unsuitable entries with `python manage_feedback.py reject ID --note "reason"`.
4. Ensure the original source CSVs are present in `data/` (they are intentionally excluded from this ZIP).
5. After the minimum of 10 approved labels for each model is reached, the next approval automatically builds `models_candidate.json`. You can also run `python train_models.py --feedback-db feedback.sqlite3 --output models_candidate.json` manually. Only approved labels are merged into training data.
6. Review the candidate's holdout metrics and test it on a separate set of fresh examples. Do not assume a higher score proves better real-world fact-checking.
7. Back up `models.json`, then replace it with `models_candidate.json` only after review. Publish the reviewed model file with the website.

The current `run_server.py` API is local-only. It is suitable for a classroom demo on the same computer, not for public internet hosting. To collect shared feedback from GitHub Pages visitors, deploy a separate backend with persistent storage and set `window.SIGNAL_CHECK_API_BASE` in `config.js` to that backend's HTTPS URL. Configure the backend's allowed origin to your exact GitHub Pages origin.
