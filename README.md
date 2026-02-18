
# Evaluating Stability Causes

| # | Potential Cause | Experiment | Results |
|---:|---|---|---|
| 0 | Instable domain | Maybe our domain is actually instable. | -- |
| 1 | Not Enough Data | Build Trees with 90% of the data 20 times. Compare the stability and performance with trees built from 10% of the data. | [sheet](https://docs.google.com/spreadsheets/d/1QwbnlBbzZGor8VxppxGtOTyznE6Vc2d_uHY9XoZ4xs4/edit?usp=sharing) |
| 2 | Sensitivity to labels | Build Trees with labels, based on different samplers (random, near, Xploit). | [sheet](https://docs.google.com/spreadsheets/d/1_Zm4IXEnXuRufCse8m9clsMzehSezMkINGUjxNLcX1c/edit?usp=sharing) |
| 3 | Model Complexity 1 | Build Trees with different tree depths. | [sheet](https://docs.google.com/spreadsheets/d/1fOnsTLJJRyzxNXbkhFF-OGm2qLv34Yt7YtqzSV_mt8s/edit?usp=sharing) |
| 4 | Model Complexity 2 | Build Trees with binary splits or multiple splits. | TBD |
| 5 | Landscape 1 | Compare the Stability of Trees with incremental clustering methods. | [sheet](https://docs.google.com/spreadsheets/d/1_7vIlcXhXdVfYhEKzGAHLbo9wbgV-QuVnf3RBY9kZ2c/edit?usp=sharing)
| 6 | Landscape 2 | Comparing stability through different datasets with different d2h distributions. | [sheet](https://docs.google.com/spreadsheets/d/1VXA_FA3PhrYczAdjauItpy6caUK53J95bqEEnwz8qV4/edit?usp=sharing) |
| 7 | Numerical methods | Compare stability through trees with Log and without Log splits. | TBD |
| 8 | Causality 1 | Apply confounder filtering to see if it helps trees. | TBD |
| 9 | Causality 2 | Compare the stability of trees with corr splits and the ones with causal splits. | TBD |

---

## **Evaluation Metrics**


### 🟦 Stability Metrics
Measures the consistency of model predictions across 20 trees for each treatment and 100 randomly selected test samples. In this measurement, we don't care about the performance at all. 

| Metric         | Description  |
|----------------|--------------|
| **Agreement** | Out of 100 randomly selected test points, how many times do all 20 trees for a given treatment make consistent predictions? <br><sup>( Sd of predictions are lower than a given threshhold. Threshhold = 0.35 * b4.sd )</sup> |
| **Comparison** | How often each treatment produces *statistically more stable* predictions compared to the others. |

---


### 🟩 Performance Metrics
Measures how accurately different models identify the optimal solutions. We split the data into two equal halves, train the models on one, and perform the optimization task on the other. 

| Metric     | Description  |
|------------|--------------|
| **Error**  | Calculates RMSE between the model’s recommended best solution and the referenced optimal found on the holdout set. |
| **Comparison** | How often a model finds a *statistically better solution* compared to other models. |

---

### Charts

- **Per dataset:** create a chart with performance and stability measures
- **Aggregation:** create a single chart for performance and a single chart for stability, both aggregating the results on all datasets.

### Planning for Feb. 18th
- Lunxiao
  - Experiment 1 : Adding a Summary of charts
  - Experiment 5 : Coding a clustering method, to use instead of Tree (comparing label + tree with label + clustering)
- Amirali
  - Experiment 2 : Adding "Adopt"
  - Experiment 6 : Data distribution analysis
  
- Things to do in general
  - Experiment 4 : Coding multiple splits into EZR Tree
  - Experiment 7 : Coding Gini index instead of Log in EZR's div() function
  - Reading about stability in SE, and start working on literature review
  
