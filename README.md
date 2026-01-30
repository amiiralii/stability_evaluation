
# Evaluating Stability Causes

| Potential Cause | Experiment | Results |
|---|---|---|
| Not Enough Data | Build Trees with 90% of the data 20 times. Compare the stability and performance with trees built from 10% of the data. | TBD |
| Sensitivity to labels | Build Trees with labels, based on different samplers (random, near, Xploit). | TBD |
| Model Complexity 1 | Build Trees with different tree depths. | TBD |
| Model Complexity 2 | Build Trees with binary splits or multiple splits. | TBD |
| Landscape 1 | Compare the Stability of Trees with incremental clustering methods. | TBD |
| Landscape 2 | Comparing stability through different datasets with different d2h distributions. | TBD |
| Numerical methods | Compare stability through trees with Log and without Log splits. | TBD |
| Causality 1 | Apply confounder filtering to see if it helps trees. | TBD |
| Causality 2 | Compare the stability of trees with corr splits and the ones with causal splits. | TBD |

### Measures

- **Stability Measure:** Consistency of predicted performance across 100 random points (in what portion of those random points do Trees have agreement)
- **Performance Measure:** MSE of the suggested optimal points from different Trees, to the actual optimal (optionally normalized)

### Charts

- **Per dataset:** create a chart with performance and stability measures

### Planning for Feb. 4th
 - **Not Enough Data** --> Lunxia
 - **Sensitivity to labels** --> Amirali
