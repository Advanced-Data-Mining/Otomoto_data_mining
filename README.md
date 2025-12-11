# Otomoto Data Mining

We want to scrape car advertisements and build models capable of predicting price based on car details.

## Data Storage

Data is saved as Parquet files in the `data/` directory, with each page of results stored as `page_XXX.parquet` for efficient storage and analysis.

## Results

- classes per step:
  - `14` cls `->` `20_000` step,
  - `10` cls `->` `30_000` step,
  - `7` cls `->` `40_000` step,
- `min_price=20_000`
- `max_price=300_000`


| Model                      | Linear bins step* | Accuracy |
|----------------------------|-------------------|----------|
| DecisionTree + TFIDF       | 20_000            | 40%      |
| DecisionTree + TFIDF       | 30_000            | 46%      |
| DecisionTree + TFIDF       | 40_000            | 52%      |
| LogisticRegression + TFIDF | 20_000            | 49%      |
| LogisticRegression + TFIDF | 30_000            | 55%      |
| LogisticRegression + TFIDF | 40_000            | 63%      |
| RF + TFIDF (only descr)    | 20_000            | 51%       |
| RF + TFIDF (only descr)    | 30_000            | 55%       |
| RF + TFIDF (only descr)    | 40_000            | 62%       |
| RFECV (num only)           | 20_000            | 52%       |
| RFECV (num only)           | 30_000            | 59%       |
| RFECV (num only)           | 40_000            | 71%       |
| RF + RFECV + TFIDF (descr + num) | 20_000            | 53%       |
| RF + RFECV + TFIDF (descr + num) | 30_000            | 59%      |
| RF + RFECV + TFIDF (descr + num) | 40_000            | 68%      |
| HerBERT fine-tuned         | 20_000            | 56%      |
| HerBert (Near-miss=1)       | 20_000             | **85%**       |
<!-- | HerBERT fine-tuned         | 30_000            | --       |
| HerBERT fine-tuned         | 40_000            | --       | -->

**Above steps are for the linear bins. Finally, in the models logarithmic split has been chosen but basing number of bins from linear one.*
