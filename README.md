
# Evaluating Stability Causes

| # | Potential Cause | Experiment | Results |
|---:|---|---|---|
| 0 | Instable domain | Maybe our domain is actually instable. | -- |
| 1 | Not Enough Data | Build Trees with 90% of the data 20 times. Compare the stability and performance with trees built from 10% of the data. | TBD |
| 2 | Sensitivity to labels | Build Trees with labels, based on different samplers (random, near, Xploit). | [Perf.](https://github.com/amiiralii/stability_evaluation/blob/main/results/2/aggregates/performance.png) & [Stab.](https://github.com/amiiralii/stability_evaluation/blob/main/results/2/aggregates/stability.png) |
| 3 | Model Complexity 1 | Build Trees with different tree depths. | TBD |
| 4 | Model Complexity 2 | Build Trees with binary splits or multiple splits. | TBD |
| 5 | Landscape 1 | Compare the Stability of Trees with incremental clustering methods. | TBD |
| 6 | Landscape 2 | Comparing stability through different datasets with different d2h distributions. | TBD |
| 7 | Numerical methods | Compare stability through trees with Log and without Log splits. | TBD |
| 8 | Causality 1 | Apply confounder filtering to see if it helps trees. | TBD |
| 9 | Causality 2 | Compare the stability of trees with corr splits and the ones with causal splits. | TBD |

### Measures

- **Stability Measure:** Consistency of predicted performance across 100 random points (in what portion of those random points do Trees have agreement)
- **Performance Measure:** MSE of the suggested optimal points from different Trees, to the actual optimal (optionally normalized)

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
