
# Evaluating Stability Causes

| # | Potential Cause | Experiment | Results |
|---:|---|---|---|
| 0 | Instable domain | Maybe our domain is actually instable. | -- |
| 1 | Not Enough Data | Build Trees with 90% of the data 20 times. Compare the stability and performance with trees built from 10% of the data. | [sheet](https://docs.google.com/spreadsheets/d/1QwbnlBbzZGor8VxppxGtOTyznE6Vc2d_uHY9XoZ4xs4/edit?usp=sharing) |
| 2 | Sensitivity to labels | Build Trees with labels, based on different samplers (random, near, Xploit). | [sheet](https://docs.google.com/spreadsheets/d/1_Zm4IXEnXuRufCse8m9clsMzehSezMkINGUjxNLcX1c/edit?usp=sharing) |
| 3 | Model Complexity 1 | Build Trees with different tree depths. | [sheet](https://docs.google.com/spreadsheets/d/1fOnsTLJJRyzxNXbkhFF-OGm2qLv34Yt7YtqzSV_mt8s/edit?usp=sharing) |
| 4 | Model Complexity 2 | Build Trees with binary splits or multiple splits. | TBD |
| 5 | Landscape 1 | Compare the Stability of Trees with incremental clustering methods. | TBD |
| 6 | Landscape 2 | Comparing stability through different datasets with different d2h distributions. | TBD |
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

### Planning for Feb. 11th
- Lunxiao
  - Experiment 1 : Executing on all datasets. 
  - Experiment 1 : Running aggregation codes and summerize in a chart. (aggregate_results.py and excel charts)
  - Experiment 1 : Show Statistics (follow below)
  - Experiment 3 : Start Running it (in case you got extra time) (experiment very similar to previous one, hyper parameter change the.leaf)
- Amirali
  - Experiment 2 : Add Statistics on performance and stability (Scott-knot)
  - Experiment 4 : Start Running it
  - Look into results and find out potential bugs(0 performance, 0 stability)
