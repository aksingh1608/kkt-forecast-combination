# Feature columns (code is the truth)

This file lists the **57** columns emitted by
`src.features.engineering.build_supervised_frame` under the default
`FeatureConfig` in `src/config.py`. Do **not** edit FeatureConfig to
match prose; rewrite the paper methods paragraph from this list.

Column names below use the target placeholder as produced for ETTh1
(`OT`). On other datasets the lag/rolling prefix is the target column
name (e.g. `Global_active_power_lag_1` on UCI). Calendar and Fourier
names are dataset-invariant.

`FeatureConfig` defaults used: `lags=(1, 2, 3, 6, 12, 24, 48, 168)`,
`rolling_windows=(3, 6, 12, 24, 48, 168)`,
`fourier_orders=((24, 3), (168, 2))`, `add_calendar=True`.

| Family | Count |
|--------|------:|
| Calendar integers | 9 |
| Calendar sin/cos | 6 |
| Autoregressive lags | 8 |
| Rolling statistics | 24 |
| Fourier terms | 10 |
| **Total** | **57** |

## Calendar integers (9)

- `hour`
- `day_of_week`
- `day_of_month`
- `day_of_year`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_weekend`

## Calendar sin/cos (6)

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `month_sin`
- `month_cos`

## Autoregressive lags (8)

- `OT_lag_1`
- `OT_lag_2`
- `OT_lag_3`
- `OT_lag_6`
- `OT_lag_12`
- `OT_lag_24`
- `OT_lag_48`
- `OT_lag_168`

## Rolling statistics (mean / std / min / max on strictly lagged values) (24)

- `OT_roll_mean_3`
- `OT_roll_std_3`
- `OT_roll_min_3`
- `OT_roll_max_3`
- `OT_roll_mean_6`
- `OT_roll_std_6`
- `OT_roll_min_6`
- `OT_roll_max_6`
- `OT_roll_mean_12`
- `OT_roll_std_12`
- `OT_roll_min_12`
- `OT_roll_max_12`
- `OT_roll_mean_24`
- `OT_roll_std_24`
- `OT_roll_min_24`
- `OT_roll_max_24`
- `OT_roll_mean_48`
- `OT_roll_std_48`
- `OT_roll_min_48`
- `OT_roll_max_48`
- `OT_roll_mean_168`
- `OT_roll_std_168`
- `OT_roll_min_168`
- `OT_roll_max_168`

## Fourier terms (10)

- `fourier_sin_24_1`
- `fourier_cos_24_1`
- `fourier_sin_24_2`
- `fourier_cos_24_2`
- `fourier_sin_24_3`
- `fourier_cos_24_3`
- `fourier_sin_168_1`
- `fourier_cos_168_1`
- `fourier_sin_168_2`
- `fourier_cos_168_2`

## Full ordered list (as in `X.columns`)

 1. `hour`
 2. `day_of_week`
 3. `day_of_month`
 4. `day_of_year`
 5. `week_of_year`
 6. `month`
 7. `quarter`
 8. `year`
 9. `is_weekend`
10. `hour_sin`
11. `hour_cos`
12. `dow_sin`
13. `dow_cos`
14. `month_sin`
15. `month_cos`
16. `OT_lag_1`
17. `OT_lag_2`
18. `OT_lag_3`
19. `OT_lag_6`
20. `OT_lag_12`
21. `OT_lag_24`
22. `OT_lag_48`
23. `OT_lag_168`
24. `OT_roll_mean_3`
25. `OT_roll_std_3`
26. `OT_roll_min_3`
27. `OT_roll_max_3`
28. `OT_roll_mean_6`
29. `OT_roll_std_6`
30. `OT_roll_min_6`
31. `OT_roll_max_6`
32. `OT_roll_mean_12`
33. `OT_roll_std_12`
34. `OT_roll_min_12`
35. `OT_roll_max_12`
36. `OT_roll_mean_24`
37. `OT_roll_std_24`
38. `OT_roll_min_24`
39. `OT_roll_max_24`
40. `OT_roll_mean_48`
41. `OT_roll_std_48`
42. `OT_roll_min_48`
43. `OT_roll_max_48`
44. `OT_roll_mean_168`
45. `OT_roll_std_168`
46. `OT_roll_min_168`
47. `OT_roll_max_168`
48. `fourier_sin_24_1`
49. `fourier_cos_24_1`
50. `fourier_sin_24_2`
51. `fourier_cos_24_2`
52. `fourier_sin_24_3`
53. `fourier_cos_24_3`
54. `fourier_sin_168_1`
55. `fourier_cos_168_1`
56. `fourier_sin_168_2`
57. `fourier_cos_168_2`
