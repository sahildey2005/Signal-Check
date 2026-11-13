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

Headlines are analyzed locally in the browser. Search terms are sent to the search provider only if a coverage link is opened.


## Developer checks

Run the dependency-free regression tests before changing the trainer:

```text
python -m py_compile train_models.py
python -m unittest discover -s tests -v
```

The test workflow also runs these checks on Python 3.10 and 3.12. The trainer reports holdout accuracy, balanced accuracy, per-class precision/recall/F1, macro F1, specificity, Brier score, and log loss. These metrics still describe only the supplied holdout split; they are not evidence of real-world fact-checking accuracy.

The optimized trainer keeps the existing browser model fields. It makes two feature-extraction passes during training to avoid storing every headline's sparse features in memory, reuses calibration/holdout scores, and stops calibration early when its gradients converge.
