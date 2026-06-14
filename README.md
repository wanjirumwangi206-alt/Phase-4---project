# Movie Recommender System

**Mountain View Streaming Platform** — personalised Top-5 recommendations for returning users

## 1. Business Understanding

### Background
Mountain view streaming platform wants to reduce churn by personalising the home screen for returning users who have rated at least 10 movies. Users are more likely to leave the platform when the recommended content does not match their preference.

### Problem Statement
Given a user's historical movie ratings, predict which unseen movies they are most likely to enjoy, and surface the Top-5 as personalised recommendations.

### Business objectives

1.Increase user retention by improving the relevance of movie recommendations shown to returning users

2.Increase User engagement

3.Balance popular content with personalised recommendations

4.Support new and returning Users


### Success Criteria
> SVD outperforms KNNBasic

> Hyperparameter tuning improves SVD

>Generate Top-5 personalized recommendations

>Address cold-start users

| Metric | Target | Why it matters |
|---|---|---|
| RMSE | < 0.90 | Predicted ratings are close to actual; penalises large errors heavily |
| MAE | < 0.70 | Average error is small and easy to interpret |
| Precision@5 | > 0.70 | ≥ 70% of surfaced movies match user preferences |
| Recall@5 | > 0.40 | System discovers ≥ 40% of movies the user would have liked |

## Summary of the data
Rating has 32 million count, 

the mean rating is 3.5,

25% percentile is 3 while the 75% percentile is 5

The maximum rating is 5 while the minimum is 0.5.

Rating csv contains floating and intergers.

Movie csv contain intergers and object

 ## 2.Exploratory Data Analysis
 Most ratings cluster between 3.0–4.5, indicating a positive bias

 Distributions are heavily right-skewed — a small number of 'power users and blockbuster movies dominate. This is the long-tail problem.'

 Drama+Romance and Comedy+Romance are the most frequent genre pairings

 ## 3.Model Comparison  KNNBasic vs SVD 

| Model | Type | Description |
|---|---|---|
| KNNBasic | Memory-based | Finds similar items via cosine similarity; interpretable but slow at scale |
| SVD | Model-based | Matrix factorisation; fast, scalable, handles sparsity well |

## 4.Hyperparameter Tuning with GridSearchCV on SVD

## 5. Evaluation: Precision@K & Recall@K

RMSE and MAE measure rating accuracy, but for a Top-5 recommendation task, **Precision@K** and **Recall@K** are more business-relevant:

- **Precision@K** — of the Top-K movies recommended, what fraction did the user actually like?
- **Recall@K** — of all movies the user would have liked, what fraction did we surface in the Top-K?

A model can have low RMSE yet still rank irrelevant movies in the top 5.

##  6. Cold-Start Problem & Hybrid Solution

| Scenario | Problem | Impact |
|---|---|---|
| New User | No ratings → no user profile | Collaborative filtering returns only global averages |
| New Movie | No ratings → no item embedding | Movie never appears in recommendations |

**Strategy:** when a user has fewer than 5 ratings, fall back to **content-based filtering** (TF-IDF on genres). Once sufficient ratings accumulate, switch to the trained SVD model.
 
 ## 7. Conclusion

### Summary
This project built a hybrid movie recommendation system on the MovieLens dataset. Two collaborative filtering algorithms (KNNBasic, SVD) were benchmarked with 5-fold cross-validation. SVD was selected for hyperparameter tuning due to its balance of accuracy and scalability.

GridSearchCV improved RMSE and MAE over the baseline. Evaluation was extended to Precision@K and Recall@K, which more directly measure recommendation list quality for the Top-5 use case.

A content-based TF-IDF genre similarity model serves as a cold-start fallback, enabling the system to serve new users before sufficient rating history has accumulated.

### Key Findings
-SVD outperforms KNNBasic on both RMSE and MAE, confirming that matrix factorisation handles sparsity better than memory-based methods.
- Recall@5 (0.67) exceeds the 0.40 target meaning the system surfaces a high proportion of movies the user would have liked.
- RMSE(0.97) and MAE (0.75) did not meet the < 0.90 / < 0.70 targets. This is attributable to the 100K sample constraint — training on the full 32M ratings would be expected to close this gap.
- Precision@5 ( 0.15 ) is below the 0.70 target, reflecting a known challenge with implicit popularity bias in sparse collaborative filtering. Re-ranking strategies or hybrid boosting would help.
- The hybrid approach extends coverage to cold-start users without sacrificing quality for returning ones.
- Drama and Comedy dominate the data genre filtering lets the platform surface diversity.

### Limitations & Next Steps
- Sample size training on the full 32M-rating dataset is expected to significantly improve RMSE, MAE, and Precision@5.
- Content-based fallback currently relies only on genres richer metadata (cast, directors, plot keywords) would improve cold-start quality.
- Precision@K re-ranking the Top-K list using diversity or popularity penalties could push Precision@5 toward the business target.
- Scale SVD is computationally expensive at full scale; production deployment would benefit from ALS or approximate nearest-neighbour methods.
-Recency ratings are static a production system should retrain incrementally as new ratings arrive.


