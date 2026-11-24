# Phase 7: Mandatory Experiment 6 - Sensitivity Study Results
**Date/Time:** 2026-11-17 16:53:46
**Context:** Scenario B Constrained Capacity (Mean Capacity = 1.549 techs/cycle, 100 slots, H=1000 cycles, 20 Evaluation Seeds)

---

## 1. Sweep 1: Sensitivity to Spare Part Lead Time $L \in \{4, 8, 12, 16\}$ cycles
*Parameters:* $C_{fail} = \$10,000$, $C_{pm} = \$2,000$, $C_{part} = \$300$, $C_{down} = \$100/\text{cycle}$.
*Fairness Enforcement:* Naive $\tau^*$ and Oracle $m^*$ re-tuned on the Tuning Pool for each lead time setting.

### Setting: Lead Time $L = 4$ cycles (Tuned: Naive $\tau^*=50$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $7,049,620 | $72,134 | 281.45 | 41.51% | 1.092 | 0.919 | $+5,543,900 [$+5,514,556, $+5,573,244] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $1,988,660 | $12,640 | 5.80 | 1.80% | 0.998 | 0.997 | $+482,940 [$+475,373, $+490,507] |
| Naive Threshold (tau=50) | $1,505,720 | $11,904 | 0.00 | 1.35% | 0.769 | 0.769 | Baseline (Ref) |
| Agent-Point | $4,380,145 | $1,699,887 | 181.55 | 20.19% | 0.977 | 0.842 | $+2,874,425 [$+2,111,206, $+3,637,644] |
| Agent-Interval | $1,325,375 | $12,576 | 0.15 | 1.09% | 0.680 | 0.680 | $-180,345 [$-186,423, $-174,267] |
| Diagnostic Oracle (m=40) | $1,433,745 | $11,491 | 0.00 | 1.29% | 0.732 | 0.732 | $-71,975 [$-76,714, $-67,236] |

### Setting: Lead Time $L = 8$ cycles (Tuned: Naive $\tau^*=60$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $7,060,835 | $70,394 | 280.60 | 41.71% | 1.088 | 0.915 | $+5,492,565 [$+5,463,473, $+5,521,657] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $2,319,610 | $1,028,099 | 30.00 | 4.08% | 1.010 | 0.992 | $+751,340 [$+288,983, $+1,213,697] |
| Naive Threshold (tau=60) | $1,568,270 | $11,918 | 0.00 | 1.42% | 0.801 | 0.800 | Baseline (Ref) |
| Agent-Point | $3,649,615 | $1,692,917 | 147.15 | 15.04% | 0.921 | 0.807 | $+2,081,345 [$+1,322,394, $+2,840,296] |
| Agent-Interval | $1,335,480 | $11,848 | 0.00 | 1.10% | 0.685 | 0.685 | $-232,790 [$-237,594, $-227,986] |
| Diagnostic Oracle (m=40) | $1,434,540 | $11,531 | 0.00 | 1.29% | 0.732 | 0.732 | $-133,730 [$-138,755, $-128,705] |

### Setting: Lead Time $L = 12$ cycles (Tuned: Naive $\tau^*=65$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $7,071,390 | $71,768 | 279.60 | 41.92% | 1.084 | 0.911 | $+5,494,030 [$+5,463,840, $+5,524,220] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $2,607,770 | $1,469,000 | 50.15 | 6.19% | 1.014 | 0.987 | $+1,030,410 [$+371,367, $+1,689,453] |
| Naive Threshold (tau=65) | $1,577,360 | $10,536 | 0.05 | 1.42% | 0.805 | 0.804 | Baseline (Ref) |
| Agent-Point | $3,181,225 | $1,553,177 | 121.85 | 11.87% | 0.884 | 0.787 | $+1,603,865 [$+905,135, $+2,302,595] |
| Agent-Interval | $1,368,095 | $13,638 | 0.05 | 1.13% | 0.701 | 0.701 | $-209,265 [$-213,784, $-204,746] |
| Diagnostic Oracle (m=40) | $1,435,200 | $11,555 | 0.00 | 1.29% | 0.732 | 0.732 | $-142,160 [$-147,765, $-136,555] |

### Setting: Lead Time $L = 16$ cycles (Tuned: Naive $\tau^*=70$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $7,080,290 | $76,275 | 278.35 | 42.13% | 1.080 | 0.907 | $+5,454,625 [$+5,421,888, $+5,487,362] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $2,974,135 | $1,732,259 | 74.75 | 8.79% | 1.027 | 0.983 | $+1,348,470 [$+569,143, $+2,127,797] |
| Naive Threshold (tau=70) | $1,625,665 | $18,482 | 0.15 | 1.46% | 0.829 | 0.829 | Baseline (Ref) |
| Agent-Point | $3,445,120 | $1,433,084 | 138.90 | 13.12% | 0.929 | 0.804 | $+1,819,455 [$+1,173,225, $+2,465,685] |
| Agent-Interval | $1,424,455 | $15,225 | 0.40 | 1.17% | 0.729 | 0.729 | $-201,210 [$-208,036, $-194,384] |
| Diagnostic Oracle (m=40) | $1,435,935 | $11,436 | 0.00 | 1.29% | 0.732 | 0.732 | $-189,730 [$-198,930, $-180,530] |

---

## 2. Sweep 2: Sensitivity to In-Service Failure Penalty $C_{fail} \in \{\$10k, \$20k, \$40k\}$
*Parameters:* Spare Lead Time $L = 8$ cycles, $C_{pm} = \$2,000$, $C_{part} = \$300$, $C_{down} = \$100/\text{cycle}$.
*Fairness Enforcement:* Naive $\tau^*$ and Oracle $m^*$ re-tuned on Tuning Pool; Cost Agent optimizes with setting $C_{fail}$.

### Setting: Failure Penalty $C_{fail} = \$10,000$ (Tuned: Naive $\tau^*=60$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $7,060,835 | $70,394 | 280.60 | 41.71% | 1.088 | 0.915 | $+5,492,565 [$+5,463,473, $+5,521,657] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $2,319,610 | $1,028,099 | 30.00 | 4.08% | 1.010 | 0.992 | $+751,340 [$+288,983, $+1,213,697] |
| Naive Threshold (tau=60) | $1,568,270 | $11,918 | 0.00 | 1.42% | 0.801 | 0.800 | Baseline (Ref) |
| Agent-Point | $3,649,615 | $1,692,917 | 147.15 | 15.04% | 0.921 | 0.807 | $+2,081,345 [$+1,322,394, $+2,840,296] |
| Agent-Interval | $1,335,480 | $11,848 | 0.00 | 1.10% | 0.685 | 0.685 | $-232,790 [$-237,594, $-227,986] |
| Diagnostic Oracle (m=40) | $1,434,540 | $11,531 | 0.00 | 1.29% | 0.732 | 0.732 | $-133,730 [$-138,755, $-128,705] |

### Setting: Failure Penalty $C_{fail} = \$20,000$ (Tuned: Naive $\tau^*=60$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $9,866,835 | $87,289 | 280.60 | 41.71% | 1.088 | 0.915 | $+8,298,565 [$+8,261,742, $+8,335,388] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $2,619,610 | $1,727,972 | 30.00 | 4.08% | 1.010 | 0.992 | $+1,051,340 [$+274,286, $+1,828,394] |
| Naive Threshold (tau=60) | $1,568,270 | $11,918 | 0.00 | 1.42% | 0.801 | 0.800 | Baseline (Ref) |
| Agent-Point | $5,121,115 | $2,541,805 | 147.15 | 15.04% | 0.921 | 0.807 | $+3,552,845 [$+2,412,106, $+4,693,584] |
| Agent-Interval | $1,351,585 | $13,122 | 0.10 | 1.11% | 0.693 | 0.693 | $-216,685 [$-222,351, $-211,019] |
| Diagnostic Oracle (m=40) | $1,434,540 | $11,531 | 0.00 | 1.29% | 0.732 | 0.732 | $-133,730 [$-138,755, $-128,705] |

### Setting: Failure Penalty $C_{fail} = \$40,000$ (Tuned: Naive $\tau^*=60$, Oracle $m^*=40$)

| Policy | Mean Cost ($) | Std ($) | Failures | Downtime % | Offered Load | Utilization | Diff vs Naive (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Run-to-Failure | $15,478,835 | $130,581 | 280.60 | 41.71% | 1.088 | 0.915 | $+13,910,565 [$+13,853,954, $+13,967,176] |
| Fixed Interval (T=75) *(Infeasible $\rho > 1$)* | $3,219,610 | $3,131,761 | 30.00 | 4.08% | 1.010 | 0.992 | $+1,651,340 [$+243,069, $+3,059,611] |
| Naive Threshold (tau=60) | $1,568,270 | $11,918 | 0.00 | 1.42% | 0.801 | 0.800 | Baseline (Ref) |
| Agent-Point | $8,064,115 | $4,245,035 | 147.15 | 15.04% | 0.921 | 0.807 | $+6,495,845 [$+4,589,173, $+8,402,517] |
| Agent-Interval | $1,361,140 | $11,276 | 0.00 | 1.11% | 0.699 | 0.699 | $-207,130 [$-212,055, $-202,205] |
| Diagnostic Oracle (m=40) | $1,434,540 | $11,531 | 0.00 | 1.29% | 0.732 | 0.732 | $-133,730 [$-138,755, $-128,705] |

---

## 3. Cost Decomposition Tables (Preventive, Spare, Downtime, Failure, Total)

### Sweep 1: Lead Time Variations ($C_{fail}=\$10k$)

| Lead Time | Policy | Preventive Overhaul ($) | Spare Parts ($) | Downtime Loss ($) | Failure Penalty ($) | Total Cost ($) | Sum Check |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| L=4 | Run-to-Failure | $0 | $84,435 | $4,150,685 | $2,814,500 | $7,049,620 | PASSED |
| L=4 | Fixed Interval (T=75) *(Infeasible)* | $1,509,200 | $241,020 | $180,440 | $58,000 | $1,988,660 | PASSED |
| L=4 | Naive Threshold (tau=50) | $1,190,400 | $179,895 | $135,425 | $0 | $1,505,720 | PASSED |
| L=4 | Agent-Point | $421,800 | $123,780 | $2,019,065 | $1,815,500 | $4,380,145 | PASSED |
| L=4 | Agent-Interval | $1,050,800 | $164,160 | $108,915 | $1,500 | $1,325,375 | PASSED |
| L=4 | Diagnostic Oracle (m=40) | $1,133,100 | $171,240 | $129,405 | $0 | $1,433,745 | PASSED |
| L=8 | Run-to-Failure | $0 | $84,180 | $4,170,655 | $2,806,000 | $7,060,835 | PASSED |
| L=8 | Fixed Interval (T=75) *(Infeasible)* | $1,382,100 | $229,875 | $407,635 | $300,000 | $2,319,610 | PASSED |
| L=8 | Naive Threshold (tau=60) | $1,238,800 | $187,965 | $141,505 | $0 | $1,568,270 | PASSED |
| L=8 | Agent-Point | $541,700 | $132,540 | $1,503,875 | $1,471,500 | $3,649,615 | PASSED |
| L=8 | Agent-Interval | $1,059,800 | $165,975 | $109,705 | $0 | $1,335,480 | PASSED |
| L=8 | Diagnostic Oracle (m=40) | $1,133,100 | $172,035 | $129,405 | $0 | $1,434,540 | PASSED |
| L=12 | Run-to-Failure | $0 | $83,880 | $4,191,510 | $2,796,000 | $7,071,390 | PASSED |
| L=12 | Fixed Interval (T=75) *(Infeasible)* | $1,267,900 | $219,720 | $618,650 | $501,500 | $2,607,770 | PASSED |
| L=12 | Naive Threshold (tau=65) | $1,244,900 | $189,660 | $142,300 | $500 | $1,577,360 | PASSED |
| L=12 | Agent-Point | $636,000 | $139,785 | $1,186,940 | $1,218,500 | $3,181,225 | PASSED |
| L=12 | Agent-Interval | $1,084,700 | $170,355 | $112,540 | $500 | $1,368,095 | PASSED |
| L=12 | Diagnostic Oracle (m=40) | $1,133,100 | $172,695 | $129,405 | $0 | $1,435,200 | PASSED |
| L=16 | Run-to-Failure | $0 | $83,505 | $4,213,285 | $2,783,500 | $7,080,290 | PASSED |
| L=16 | Fixed Interval (T=75) *(Infeasible)* | $1,139,700 | $208,320 | $878,615 | $747,500 | $2,974,135 | PASSED |
| L=16 | Naive Threshold (tau=70) | $1,282,100 | $196,005 | $146,060 | $1,500 | $1,625,665 | PASSED |
| L=16 | Agent-Point | $603,900 | $140,280 | $1,311,940 | $1,389,000 | $3,445,120 | PASSED |
| L=16 | Agent-Interval | $1,126,100 | $177,615 | $116,740 | $4,000 | $1,424,455 | PASSED |
| L=16 | Diagnostic Oracle (m=40) | $1,133,100 | $173,430 | $129,405 | $0 | $1,435,935 | PASSED |

### Sweep 2: Failure Penalty Variations ($L=8$)

| Failure Penalty | Policy | Preventive Overhaul ($) | Spare Parts ($) | Downtime Loss ($) | Failure Penalty ($) | Total Cost ($) | Sum Check |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $10k | Run-to-Failure | $0 | $84,180 | $4,170,655 | $2,806,000 | $7,060,835 | PASSED |
| $10k | Fixed Interval (T=75) *(Infeasible)* | $1,382,100 | $229,875 | $407,635 | $300,000 | $2,319,610 | PASSED |
| $10k | Naive Threshold (tau=60) | $1,238,800 | $187,965 | $141,505 | $0 | $1,568,270 | PASSED |
| $10k | Agent-Point | $541,700 | $132,540 | $1,503,875 | $1,471,500 | $3,649,615 | PASSED |
| $10k | Agent-Interval | $1,059,800 | $165,975 | $109,705 | $0 | $1,335,480 | PASSED |
| $10k | Diagnostic Oracle (m=40) | $1,133,100 | $172,035 | $129,405 | $0 | $1,434,540 | PASSED |
| $20k | Run-to-Failure | $0 | $84,180 | $4,170,655 | $5,612,000 | $9,866,835 | PASSED |
| $20k | Fixed Interval (T=75) *(Infeasible)* | $1,382,100 | $229,875 | $407,635 | $600,000 | $2,619,610 | PASSED |
| $20k | Naive Threshold (tau=60) | $1,238,800 | $187,965 | $141,505 | $0 | $1,568,270 | PASSED |
| $20k | Agent-Point | $541,700 | $132,540 | $1,503,875 | $2,943,000 | $5,121,115 | PASSED |
| $20k | Agent-Interval | $1,071,300 | $167,700 | $110,585 | $2,000 | $1,351,585 | PASSED |
| $20k | Diagnostic Oracle (m=40) | $1,133,100 | $172,035 | $129,405 | $0 | $1,434,540 | PASSED |
| $40k | Run-to-Failure | $0 | $84,180 | $4,170,655 | $11,224,000 | $15,478,835 | PASSED |
| $40k | Fixed Interval (T=75) *(Infeasible)* | $1,382,100 | $229,875 | $407,635 | $1,200,000 | $3,219,610 | PASSED |
| $40k | Naive Threshold (tau=60) | $1,238,800 | $187,965 | $141,505 | $0 | $1,568,270 | PASSED |
| $40k | Agent-Point | $541,700 | $132,540 | $1,503,875 | $5,886,000 | $8,064,115 | PASSED |
| $40k | Agent-Interval | $1,080,800 | $169,095 | $111,245 | $0 | $1,361,140 | PASSED |
| $40k | Diagnostic Oracle (m=40) | $1,133,100 | $172,035 | $129,405 | $0 | $1,434,540 | PASSED |