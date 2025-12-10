# Otomoto Data Mining

We want to scrape car advertisements and build models capable of predicting price based on car details.

## Data Storage

Data is saved as Parquet files in the `data/` directory, with each page of results stored as `page_XXX.parquet` for efficient storage and analysis.

## Results

| Model                    | Linear bins step* | Accuracy |
|--------------------------|-------------------|--------|
| RF + RFECV               | 20_000            | --     |
| RF + RFECV               | 30_000            | --     |
| RF + RFECV               | 40_000            | --     |
| RF + RFECV + TFIDF       | 20_000            | --     |
| RF + RFECV + TFIDF       | 30_000            | --%    |
| RF + RFECV + TFIDF       | 40_000            | --%    |
| HerBERT fine-tuned       | 20_000            | --%    |
| HerBERT fine-tuned       | 30_000            | --%    |
| HerBERT fine-tuned       | 40_000            | --%    |
| DecisionTree + TFIDF     | 20_000            | 48%      |
| DecisionTree + TFIDF     | 30_000            | --%    |
| DecisionTree + TFIDF     | 40_000            | --%    |
*Above steps is for the linear bins. Finally, in the models logarithmic split has been chosen but basing number of bins 
from linear one. 


## TODO

### Models to try

- [x] Random Forest with RFECV.
- [x] Random Forest with RFECV and TFIDF for description.
- [x] Fine tuned [HerBERT](https://huggingface.co/docs/transformers/en/model_doc/herbert).
- [ ] Decision Tree with TFIDF description vectors.
- [ ] Ensemble: CatBoost for categorical features and XGBoost for numerical features.
- [ ] Ensemble: CatBoost for categorical features, XGBoost for numerical features and HerBERT for contextual description processing.
- [ ] Fixed [HerBERT](https://huggingface.co/docs/transformers/en/model_doc/herbert) with trained classifier layers.
- [ ] Low Rank Adaptation on HerBERT instead of fine-tuning.
- [ ] Use HerBERT as a vector embedding extraction layer + classifier (like XGB).

### Data exploration

- [ ] Verify if car model names and brands are normalized, i.e. Iveco Daily and IVECO Daily. This can be done by measuring Levenstein distance on the closest field.
