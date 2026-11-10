# Location-Aware Crop Recommender (prototype)

Pick a point on a map → fetch local climate (NASA POWER) and soil pH (ISRIC SoilGrids) →
enter N, P, K → get the top-3 crops with probabilities and a SHAP explanation.

This prototype implements the gaps from the literature review: coordinate-based lookup,
region-level validation, and explainable output.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get the data
Download the Kaggle **"Crop Recommendation Dataset"** (2,200 rows, 22 crops) and save it as
`data/Crop_recommendation.csv`. Columns: `N, P, K, temperature, humidity, ph, rainfall, label`.

No Kaggle account yet? Test the pipeline with fake data first:
`python make_synthetic.py` (writes `data/synthetic_demo.csv`; SYNTHETIC, never report its results).

## 2. Train and validate
```bash
python train.py --data data/Crop_recommendation.csv
```
Trains Random Forest (and XGBoost if installed), prints accuracy/precision/recall/F1 under 5-fold CV,
and saves `models/crop_model.joblib`.

**Region-level validation (the gap in the review).** The Kaggle file has no region column, so random
CV cannot test unseen regions. If you build or find a dataset with a region/zone/district column:
```bash
python train.py --data mydata.csv --group-col region
```
This adds leave-region-out validation, where whole regions are held out. Report both numbers and
expect the second to be lower.

## 3. Run the app
```bash
streamlit run app.py
```
Click the map (or type coordinates) → choose growing-season months → **Fetch** → edit values if needed → **Recommend**.

## Files
| File | Purpose |
|---|---|
| `train.py` | Training + validation (random and leave-region-out) |
| `location.py` | Coordinates → temperature, humidity, rainfall (NASA POWER), soil pH (SoilGrids) |
| `model_utils.py` | Top-k prediction and SHAP helpers |
| `app.py` | Streamlit interface |
| `make_synthetic.py` | Fake data for pipeline testing only |

## Known limitations (state these in your report)
- **Location acts only through the lookup.** The Kaggle data has no coordinates, so the model never learns geography.
- **Unit/definition mismatch.** The dataset's rainfall is treated as mean mm per month over the season, and humidity/temperature as seasonal means. This is an assumption; document it.
- **N, P, K are entered by the user.** Public global soil APIs do not provide them in the dataset's units. Use a soil-health card or district averages.
- **Climate is a long-term average**, not a forecast.
- **Benchmark data is clean**, so very high accuracy says little about real farms. Test on several real locations and compare with local agricultural guidance.

## Not yet verified
The API calls in `location.py` were written from documentation and tested only against mock responses.
Run one fetch on your machine; if an API's format has changed, adjust `parse_power` / `parse_soil_ph`.
`app.py` was syntax-checked but not run in the authoring environment.

## Ideas to extend (for the paper's future-work section)
Elevation and agro-climatic-zone features, district-level soil data, weather-forecast integration
(SARIMAX/Open-Meteo), spatial cross-validation on multi-region data, local-language output.
