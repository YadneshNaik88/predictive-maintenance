# Phase 5d: Consolidated Maintenance Simulation Results
**Generated:** 2026-11-16 14:06:44 UTC
**Simulator Model:** Recalibrated Discrete-Event Technician Queue with Strict Conservation

## 1. Capacity Schedule Recalibration & Validity Checks Summary (Tuning Pool)
- **Scenario B (Target Utilization: 0.75 - 0.85):**
  - Capacity Schedule: Values [0, 1, 2, 3], Probabilities [0.1106, 0.3594, 0.3894, 0.1406] (Mean capacity = 1.549 techs/cycle, Seed 42).
  - Tuned Naive Threshold Winner: tau* = 60
  - Validity Check 1 (Offered Load < 1.0): PASSED (0.799)
  - Validity Check 2 (Total Downtime < 10% of slot-cycles): PASSED (1,414c / 1.41%)
  - Validity Check 3 (Winning tau not on edge < 80): PASSED (tau* = 60 < 80)
  - Achieved Utilization: PASSED (0.798 on Tuning, 0.800 on Evaluation, target 0.75-0.85)
- **Scenario C (Target Utilization: 0.88 - 0.92):**
  - Capacity Schedule: Values [0, 1, 2, 3], Probabilities [0.1250, 0.3750, 0.3750, 0.1250] (Mean capacity = 1.487 techs/cycle, Seed 42).
  - Tuned Naive Threshold Winner: tau* = 70
  - Validity Check 1 (Offered Load < 1.0): PASSED (0.893)
  - Validity Check 2 (Total Downtime < 10% of slot-cycles): PASSED (1,538c / 1.54%)
  - Validity Check 3 (Winning tau not on edge < 80): PASSED (tau* = 70 < 80)
  - Achieved Utilization: PASSED (0.892 on Tuning, 0.900 on Evaluation, target 0.88-0.92)

## 2. Hyperparameter Tuning Results (Tuning Pool: 50 Engines, 20 Seeds)
- **Selection Rule:** Lowest mean cost; if another candidate is within one standard deviation and has fewer failures, prefer it.
- **Scenario A (Abundant):** $T^* = 125$, $\tau^* = 15$, $m^* = 2$
- **Scenario B (Constrained):** $T^* = 75$, $\tau^* = 60$, $m^* = 40$
- **Scenario C (Tight):** $T^* = 75$, $\tau^* = 70$, $m^* = 40$

## 3. Evaluation Benchmark Results (Evaluation Pool: 50 Engines, 20 Seeds)

> [!NOTE]
> **Diagnostic Oracle Reference Bound:** The diagnostic oracle triggers spare ordering based strictly on true RUL ($RUL \le L_{\text{spare}} + m$) and ignores the dynamic technician availability schedule. In constrained queueing environments, an unbuffered oracle can experience queue backlogs and failure cascades. When appropriately queue-buffered ($m^*=40$), it acts as a reliable theoretical upper bound.

### Scenario A: Abundant Capacity (Constant 4 Technicians/Cycle)
| Policy | Total Cost | Cost Floor | % Above Floor | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead (cycles) | Mean Q-Wait (cycles) | Utilization | Offered Load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run-to-Failure | $5,111,475 +/- $53,210 | $1,675,875 +/- $30,157 | 205.1% +/- 3.8% | 429.45 +/- 4.52 | 0.0 +/- 0.0 | 6881 (6.88%) +/- 290 | 11.16 +/- 0.71 | 3.16 +/- 0.71 | 0.636 +/- 0.007 | 0.644 +/- 0.007 |
| Fixed-Interval(T=125) | $1,999,810 +/- $492 | $1,999,795 +/- $556 | 0.0% +/- 0.0% | 0.00 +/- 0.00 | 800.0 +/- 0.2 | 1599 (1.60%) +/- 1 | 8.01 +/- 0.01 | 0.01 +/- 0.01 | 0.400 +/- 0.000 | 0.400 +/- 0.000 |
| Naive-Threshold(tau=15) | $1,193,155 +/- $12,459 | $1,191,820 +/- $12,545 | 0.1% +/- 0.1% | 0.00 +/- 0.00 | 476.8 +/- 5.0 | 953 (0.95%) +/- 10 | 8.01 +/- 0.01 | 0.01 +/- 0.01 | 0.238 +/- 0.003 | 0.238 +/- 0.003 |
| Diagnostic-Oracle(margin=2) | $1,159,970 +/- $11,327 | $1,158,695 +/- $11,731 | 0.1% +/- 0.1% | 0.00 +/- 0.00 | 463.5 +/- 4.7 | 926 (0.93%) +/- 9 | 8.01 +/- 0.01 | 0.01 +/- 0.01 | 0.232 +/- 0.002 | 0.232 +/- 0.002 |

- **Diagnostic Oracle Headroom over Naive Threshold:** 2.78%
- **Share of Achievable Savings Captured by Naive:** 99.16%
- **Exp 3 (Run-to-Failure vs Fixed-Interval):** Fixed achieves 60.88% cost reduction over RTF ($1,999,810 vs $5,111,475)
- **Exp 4 (Fixed-Interval vs Naive-Threshold):** Naive achieves 40.34% cost reduction over Fixed ($1,193,155 vs $1,999,810)

### Scenario B: Constrained Capacity (Schedule Mean = 1.549 Technicians/Cycle, Target Util ~75-85%)
| Policy | Total Cost | Cost Floor | % Above Floor | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead (cycles) | Mean Q-Wait (cycles) | Utilization | Offered Load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run-to-Failure | $7,060,835 +/- $70,394 | $4,816,035 +/- $61,718 | 46.6% +/- 0.6% | 280.60 +/- 2.62 | 0.0 +/- 0.0 | 41707 (41.71%) +/- 603 | 152.04 +/- 1.99 | 144.04 +/- 1.99 | 0.915 +/- 0.004 | 1.088 +/- 0.010 |
| Fixed-Interval(T=75) | $2,319,610 +/- $1,028,099 | $2,066,050 +/- $475,298 | 8.9% +/- 16.2% | 30.00 +/- 70.29 | 691.0 +/- 172.0 | 4076 (4.08%) +/- 7047 | 72.47 +/- 29.75 | 64.47 +/- 29.75 | 0.992 +/- 0.000 | 1.010 +/- 0.051 |
| Naive-Threshold(tau=60) | $1,568,270 +/- $11,918 | $1,566,125 +/- $12,050 | 0.1% +/- 0.1% | 0.00 +/- 0.00 | 619.4 +/- 4.7 | 1415 (1.42%) +/- 20 | 12.52 +/- 0.75 | 4.52 +/- 0.75 | 0.800 +/- 0.006 | 0.801 +/- 0.006 |
| Diagnostic-Oracle(margin=40) | $1,434,540 +/- $11,531 | $1,432,470 +/- $11,642 | 0.1% +/- 0.1% | 0.00 +/- 0.00 | 566.5 +/- 4.6 | 1294 (1.29%) +/- 16 | 11.36 +/- 0.47 | 3.36 +/- 0.47 | 0.732 +/- 0.006 | 0.732 +/- 0.006 |

- **Diagnostic Oracle Headroom over Naive Threshold:** 8.53%
- **Share of Achievable Savings Captured by Naive:** 97.62%
- **Exp 3 (Run-to-Failure vs Fixed-Interval):** Fixed achieves 67.15% cost reduction over RTF ($2,319,610 vs $7,060,835)
- **Exp 4 (Fixed-Interval vs Naive-Threshold):** Naive achieves 32.39% cost reduction over Fixed ($1,568,270 vs $2,319,610)

### Scenario C: Tight Capacity (Schedule Mean = 1.487 Technicians/Cycle, Target Util ~88-92%)
| Policy | Total Cost | Cost Floor | % Above Floor | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead (cycles) | Mean Q-Wait (cycles) | Utilization | Offered Load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run-to-Failure | $7,159,850 +/- $74,936 | $4,979,050 +/- $60,459 | 43.8% +/- 0.5% | 272.60 +/- 2.82 | 0.0 +/- 0.0 | 43521 (43.52%) +/- 573 | 164.46 +/- 1.97 | 156.46 +/- 1.97 | 0.914 +/- 0.004 | 1.101 +/- 0.011 |
| Fixed-Interval(T=75) | $3,722,345 +/- $1,811,914 | $2,673,790 +/- $1,031,856 | 33.6% +/- 18.9% | 129.35 +/- 100.07 | 422.1 +/- 238.2 | 14056 (14.06%) +/- 13472 | 117.91 +/- 50.84 | 109.91 +/- 50.84 | 0.991 +/- 0.000 | 1.091 +/- 0.089 |
| Naive-Threshold(tau=70) | $1,696,380 +/- $16,358 | $1,692,555 +/- $15,560 | 0.2% +/- 0.2% | 0.15 +/- 0.48 | 668.0 +/- 6.1 | 1557 (1.56%) +/- 16 | 15.35 +/- 0.58 | 7.35 +/- 0.58 | 0.900 +/- 0.009 | 0.900 +/- 0.009 |
| Diagnostic-Oracle(margin=40) | $1,429,310 +/- $11,197 | $1,427,240 +/- $11,363 | 0.1% +/- 0.1% | 0.00 +/- 0.00 | 564.0 +/- 4.6 | 1299 (1.30%) +/- 13 | 12.11 +/- 0.47 | 4.11 +/- 0.47 | 0.759 +/- 0.006 | 0.760 +/- 0.006 |

- **Diagnostic Oracle Headroom over Naive Threshold:** 15.74%
- **Share of Achievable Savings Captured by Naive:** 95.34%
- **Exp 3 (Run-to-Failure vs Fixed-Interval):** Fixed achieves 48.01% cost reduction over RTF ($3,722,345 vs $7,159,850)
- **Exp 4 (Fixed-Interval vs Naive-Threshold):** Naive achieves 54.43% cost reduction over Fixed ($1,696,380 vs $3,722,345)
