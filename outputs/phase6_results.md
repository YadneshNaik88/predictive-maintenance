# Phase 6: Multi-Agent Predictive Maintenance System Benchmark
**Generated:** 2026-11-17 10:30:00 UTC  
**Evaluation Pool:** 50 Held-Out C-MAPSS Engines, 20 Random Seeds  
**Multi-Agent Architecture:** Prognostic, Resource, Cost, and Coordinator Agents  

## Executive Summary & Key Findings
- **Mandatory Experiment 2 (Agent-Point vs Agent-Interval):** Agent-Point ignores uncertainty ($R = \text{pred}$), scheduling maintenance aggressively late and suffering catastrophic failure cascades. Across all scenarios, Agent-Interval completely eliminates or dramatically suppresses failures, reducing total financial costs by **$1.16M to $3.16M per fleet run (48.8% to 70.2% cost reduction)**.
- **Mandatory Experiment 5 (Naive-Threshold vs Agent-Interval):**
  - **Scenario A (Abundant Resources):** The tuned Naive Threshold achieves $1,193,155 +/- $12,459 while Agent-Interval achieves $1,217,655 +/- $12,048. Agent-Interval is **statistically significantly worse** by $24,500 (+2.05%, 95% CI [$+21,196, $+27,804], excluding zero), though within 2.05% without tuning any threshold parameter.
  - **Scenario B (Constrained Resources, Util ~80%):** Agent-Interval achieves **$1,335,480 +/- $11,848** (0.00 failures), beating the tuned Naive Threshold ($1,568,270) by **$232,790 (14.84% savings; 95% CI [$-237,920, $-227,660])** and beating the Diagnostic Oracle ($1,434,540) by $99,060 because the agent proactively smooths reservations across a 60-cycle lookahead window, avoiding crew congestion during low-capacity dips.
  - **Scenario C (Tight Resources, Util ~90%):** Agent-Interval achieves **$1,340,715 +/- $12,395** (0.10 failures), beating the tuned Naive Threshold ($1,696,380) by **$355,665 (20.97% savings; 95% CI [$-364,411, $-346,919])** and beating the Diagnostic Oracle ($1,429,310) by $88,595.
- **Parameter Tuning Policy:** The Agentic policy has **no parameters tuned** on the evaluation pool ($W=60$, $k=5$, 10-cycle bins, 90% confidence level are fixed design choices).
- **Information Integrity:** Strictly verified that agent modules have zero access to ground truth ($true\_rul$), future lifetime draws, or simulator random state.

---

## 1. Scenario Evaluation Tables (20 Seeds on Held-Out Evaluation Pool)

### Scenario A: Abundant Capacity (4 Technicians/Cycle, Lead Time 8)
*Cost Floor: $1,100,000*
| Policy | Total Cost (Mean +/- Std) | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead | Mean Q-Wait | Util | Offered Load | Distinct PM Escalations (% PM) | Cost vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run-to-Failure** | $5,111,475 +/- $53,210 | 429.45 | 0.00 | 6,881 (6.88%) | 11.16 | 3.16 | 0.636 | 0.644 | — | +328.4% |
| **Fixed Interval (T=125)** | $1,999,810 +/- $492 | 0.00 | 800.00 | 1,599 (1.60%) | 8.01 | 0.01 | 0.400 | 0.400 | — | +67.6% |
| **Naive Threshold (tau=15)** | $1,193,155 +/- $12,459 | 0.00 | 476.80 | 953 (0.95%) | 8.01 | 0.01 | 0.238 | 0.238 | — | 0.0% (Ref) |
| **Agent-Point** | $2,378,330 +/- $186,575 | 143.65 | 314.20 | 1,662 (1.66%) | 70.76 | 62.76 | 0.372 | 0.373 | 1.00 (0.32%) | +99.3% |
| **Agent-Interval** | **$1,217,655 +/- $12,048** | **0.00** | **483.45** | **966 (0.97%)** | **61.95** | **53.95** | **0.242** | **0.242** | **0.00 (0.00%)** | **+2.05%** |
| **Agent-Interval-Blind** | $1,217,655 +/- $12,048 | 0.00 | 483.45 | 966 (0.97%) | 61.95 | 53.95 | 0.242 | 0.242 | 0.00 (0.00%) | +2.05% |
| **Agent-Perfect (Ref Only)** | $1,162,075 +/- $10,676 | 0.00 | 464.80 | 929 (0.93%) | 66.82 | 58.98 | 0.232 | 0.232 | 0.00 (0.00%) | -2.61% |
| **Diagnostic Oracle (m=2)** | $1,159,970 +/- $11,327 | 0.00 | 463.50 | 926 (0.93%) | 8.01 | 0.01 | 0.232 | 0.232 | — | -2.78% |

---

### Scenario B: Constrained Capacity (Mean 1.549 Technicians/Cycle, Lead Time 8)
*Cost Floor: $1,100,000*
| Policy | Total Cost (Mean +/- Std) | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead | Mean Q-Wait | Util | Offered Load | Distinct PM Escalations (% PM) | Cost vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run-to-Failure** | $7,060,835 +/- $70,394 | 280.60 | 0.00 | 41,707 (41.71%) | 152.04 | 144.04 | 0.915 | 1.088 | — | +350.2% |
| **Fixed Interval (T=75)** | $2,319,610 +/- $1,028,099 | 30.00 | 691.00 | 4,076 (4.08%) | 72.47 | 64.47 | 0.992 | 1.010 | — | +47.9% |
| **Naive Threshold (tau=60)** | $1,568,270 +/- $11,918 | 0.00 | 619.40 | 1,415 (1.42%) | 12.52 | 4.52 | 0.800 | 0.801 | — | 0.0% (Ref) |
| **Agent-Point** | $3,649,615 +/- $1,692,917 | 147.15 | 270.85 | 15,039 (15.04%) | 97.78 | 89.78 | 0.807 | 0.921 | 81.55 (30.11%) | +132.7% |
| **Agent-Interval** | **$1,335,480 +/- $11,848** | **0.00** | **529.90** | **1,097 (1.10%)** | **45.68** | **37.68** | **0.685** | **0.685** | **50.20 (9.47%)** | **-14.84%** |
| **Agent-Interval-Blind** | $1,357,835 +/- $10,070 | 0.25 | 536.85 | 1,123 (1.12%) | 43.64 | 35.64 | 0.695 | 0.695 | 58.10 (10.82%) | -13.42% |
| **Agent-Perfect (Ref Only)** | $5,560,380 +/- $2,190,661 | 219.80 | 179.35 | 27,243 (27.24%) | 163.13 | 155.13 | 0.884 | 1.102 | 121.40 (67.69%) | +254.6% |
| **Diagnostic Oracle (m=40)** | $1,434,540 +/- $11,531 | 0.00 | 566.50 | 1,294 (1.29%) | 11.36 | 3.36 | 0.732 | 0.732 | — | -8.53% |

---

### Scenario C: Tight Capacity (Mean 1.487 Technicians/Cycle, Lead Time 8)
*Cost Floor: $1,100,000*
| Policy | Total Cost (Mean +/- Std) | Failures | Preventive Actions | Downtime Cycles (% slot-cycles) | Avg Lead | Mean Q-Wait | Util | Offered Load | Distinct PM Escalations (% PM) | Cost vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run-to-Failure** | $7,159,850 +/- $74,936 | 272.60 | 0.00 | 43,521 (43.52%) | 164.46 | 156.46 | 0.914 | 1.101 | — | +322.1% |
| **Fixed Interval (T=75)** | $3,722,345 +/- $1,811,914 | 129.35 | 422.10 | 14,056 (14.06%) | 117.91 | 109.91 | 0.991 | 1.091 | — | +119.4% |
| **Naive Threshold (tau=70)** | $1,696,380 +/- $16,358 | 0.15 | 668.00 | 1,557 (1.56%) | 15.35 | 7.35 | 0.900 | 0.900 | — | 0.0% (Ref) |
| **Agent-Point** | $4,495,765 +/- $1,845,122 | 178.10 | 209.50 | 21,731 (21.73%) | 122.20 | 114.20 | 0.852 | 1.002 | 74.30 (35.47%) | +165.0% |
| **Agent-Interval** | **$1,340,715 +/- $12,395** | **0.10** | **531.05** | **1,112 (1.11%)** | **45.21** | **37.21** | **0.716** | **0.716** | **63.25 (11.91%)** | **-20.97%** |
| **Agent-Interval-Blind** | $1,357,285 +/- $11,054 | 0.30 | 536.45 | 1,123 (1.12%) | 44.13 | 36.13 | 0.724 | 0.724 | 71.15 (13.26%) | -19.99% |
| **Agent-Perfect (Ref Only)** | $6,470,090 +/- $1,216,829 | 253.50 | 79.40 | 36,450 (36.45%) | 190.90 | 182.90 | 0.893 | 1.132 | 74.10 (93.32%) | +281.4% |
| **Diagnostic Oracle (m=40)** | $1,429,310 +/- $11,197 | 0.00 | 564.00 | 1,299 (1.30%) | 12.11 | 4.11 | 0.759 | 0.760 | — | -15.74% |

---

## 2. Escalation Audit & Distinct Maintenance Job Analysis
An escalation occurs inside `CoordinatorAgent.plan_fleet` when no feasible maintenance window exists in the normal candidate range $[now + L, now + \min(\text{pred}, W)]$. The Resource Agent searches forward beyond predicted RUL to find the earliest cycle where technician capacity is available, orders the spare part immediately, and reserves that future slot.
- **Strict Conservation & Fairness:**
  - An escalated job gets **zero priority over failed engines** (failed engines are always Priority 1).
  - An escalated job gets **zero priority over other planned jobs** (planned jobs are dispatched strictly by urgency lower bound).
  - An escalated job receives **no extra technicians, no cost discount, and no lead time reduction**.
- **Per-Job Escalation Statistics (Evaluation Pool, 20 seeds):**
  - **Scenario A:** Agent-Point escalates 1.00 PM job/run (0.32% of PM jobs; mean cost $2,500.00 vs $2,500.00 normal). Agent-Interval has 0 escalations.
  - **Scenario B:**
    - Agent-Point: 81.55 distinct PM jobs escalated (30.11% of PM jobs). Mean cost: $2,541.76 vs $2,519.34 normal.
    - Agent-Interval: 50.20 distinct PM jobs escalated (9.47% of PM jobs). Mean cost: $2,540.11 vs $2,503.51 normal.
  - **Scenario C:**
    - Agent-Point: 74.30 distinct PM jobs escalated (35.47% of PM jobs). Mean cost: $2,543.23 vs $2,527.00 normal.
    - Agent-Interval: 63.25 distinct PM jobs escalated (11.91% of PM jobs). Mean cost: $2,542.73 vs $2,504.83 normal.
  - *Explanation of Job Cost:* Normal PM jobs cost $2,500.00 ($2,000 overhaul + $300 spare + 2 cycles x $100 downtime). Escalated PM jobs incur slightly higher mean cost ($2,540–$2,543) because they take tighter crew slots where minor contention causes ~0.4 cycles of additional downtime.

---

## 3. Reference Investigation: Agent-Perfect vs. Agent-Interval
- In **Scenario A** (unconstrained capacity), Agent-Perfect achieves $1,162,075 (0 fails), closely matching the Oracle ($1,159,970) and beating Agent-Interval ($1,217,655).
- In **Scenarios B and C** (constrained capacity), Agent-Perfect catastrophically collapses ($5.56M in B, $6.47M in C, with >219 failures).
- **Mechanism of Collapse:**
  1. Agent-Perfect has zero uncertainty spread ($e = [0.0]$), so its expected failure risk is 0% for all $s < \text{true\_rul}$.
  2. Under renewal reward theory ($C(s)/E(s)$), zero risk forces the agent to schedule maintenance at the very last possible cycle ($s = \text{true\_rul} - 1$), leaving **zero uncertainty margin**.
  3. In Scenarios B and C, where technician capacity is stochastic and constrained (mean ~1.5 techs/cycle, with 0-capacity cycles), any microscopic queue delay causes an engine scheduled at $s = \text{true\_rul}-1$ to fail in queue.
  4. Corrective repairs require $D_{\text{fail}}=6$ cycles (3x preventive duration) and take strict priority over planned maintenance, instantly consuming available technician capacity and causing cascading queue starvation across the entire fleet.
  5. In stark contrast, **Agent-Interval incorporates the empirical residual distribution**: as an engine approaches end of life, non-zero tail risk ($P(R \le s) > 0$) penalizes late candidates, causing the agent to naturally schedule maintenance 10–20 cycles before failure. This uncertainty buffer acts as an endogenous shock absorber against technician queue delays, completely preventing the queue death spiral.

---

## 4. Ablation Study: Roster-Blind Maintenance Scheduling (Agent-Interval-Blind)
- In `Agent-Interval-Blind`, the Resource Agent does not observe the published technician roster for future cycles. It sees only the current cycle's capacity and queue, assuming future capacity equals the scenario mean $\bar{c}$.
- **Paired Comparisons (20 seeds on Evaluation Pool):**
  - **Scenario A:** Identical to Agent-Interval ($1,217,655 +/- $12,048) because Scenario A has constant capacity $\bar{c} = 4.0$.
  - **Scenario B:** Cost $1,357,835 +/- $10,070 (0.25 failures).
    - Paired difference vs Naive Threshold: **-$210,435** (95% CI: [**-$215,106, -$205,764**]).
    - Paired difference vs Agent-Interval: **+$22,355** (95% CI: [**+$19,400, +$25,310**]).
    - Roster lookahead saves $22,355 and eliminates all failures. Even without lookahead, the blind agent beats Naive Threshold by 13.4%.
  - **Scenario C:** Cost $1,357,285 +/- $11,054 (0.30 failures).
    - Paired difference vs Naive Threshold: **-$339,095** (95% CI: [**-$347,023, -$331,167**]).
    - Paired difference vs Agent-Interval: **+$16,570** (95% CI: [**+$13,807, +$19,333**]).
    - Roster lookahead saves $16,570 and suppresses failures from 0.30 to 0.10. Even without lookahead, the blind agent beats Naive Threshold by 20.0%.

---

## 5. Study Limitations & Statistical Context
1. **Evaluation Pool Sampling:** Confidence intervals reflect simulation seeds (engine renewal draws and desynchronized start ages with replacement) over a fixed 50-engine held-out evaluation pool. They do not account for variance across alternative engine splits.
2. **Information Asymmetry:** The Agentic policy utilizes a published technician roster for the next $W=60$ cycles, whereas baseline policies (Run-to-Failure, Fixed Interval, Naive Threshold, Diagnostic Oracle) do not observe future roster capacity.
3. **The Diagnostic Oracle is Not a Theoretical Lower Bound:** The Diagnostic Oracle is a condition-based policy with access to perfect RUL information but zero resource lookahead. In constrained environments, it triggers maintenance bursts during crew capacity dips, incurring queue delays. Agent-Interval outperforming the Oracle in Scenarios B and C is a direct result of proactive capacity smoothing across the lookahead window, not a mathematical anomaly.
