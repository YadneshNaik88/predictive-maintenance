Created At: 2026-11-18T16:40:29+05:30
Completed At: 2026-11-18T16:40:29+05:30
Created At: 2026-11-18T16:40:29+05:30
Completed At: 2026-11-18T16:40:29+05:30
Showing lines 1-150 of 606
1: """
2: agents.py
3: 
4: Phase 6: Multi-Agent Collaborative Predictive Maintenance System.
5: Implements the 4 collaborative agents adhering strictly to the PS-06 brief:
6: 1. Prognostic Agent (RUL predictions, uncertainty bounds, and empirical residual risk table)
7: 2. Resource Agent (spare part lead time tracking, technician roster & reservation calendar)
8: 3. Cost Agent (rate-based renewal cost optimization C(s)/E(s))
9: 4. Scheduling/Coordinator Agent (fleet-level planning pass, priority dispatch, escalation, JSONL decision logging)
10: 
11: Includes two variants for Mandatory Experiment 2:
12: - Agent-Interval (uncertainty-aware empirical residual risk distribution)
13: - Agent-Point (deterministic point-estimate baseline, s < pred_rul, R = pred_rul)
14: """
15: 
16: from dataclasses import dataclass, field
17: import json
18: from pathlib import Path
19: from typing import Any, Dict, List, Optional, Tuple, Union
20: import numpy as np
21: import pandas as pd
22: import yaml
23: 
24: from src.policies import BasePolicy
25: from src.simulator import (
26:     DEFAULT_OOF_PATH,
27:     DEFAULT_CONFIG_PATH,
28:     EngineObservation,
29:     get_tuning_eval_pools,
30: )
31: 
32: PROJECT_ROOT = Path(__file__).resolve().parents[1]
33: DEFAULT_DECISION_LOG_PATH = PROJECT_ROOT / "outputs" / "decision_log.jsonl"
34: 
35: 
36: class PrognosticAgent:
37:     """
38:     Agent 1: Prognostic Agent.
39:     Provides predicted RUL point estimates, 90% uncertainty intervals, and
40:     an empirical residual risk table built strictly from TUNING-pool out-of-fold predictions.
41:     Computes P(R <= s) and E[min(s, R)] where R = pred_rul + e.
42:     """
43: 
44:     def __init__(
45:         self,
46:         oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH,
47:         tuning_units: Optional[np.ndarray] = None,
48:         use_intervals: bool = True,
49:     ) -> None:
50:         self.use_intervals = use_intervals
51:         if tuning_units is None:
52:             tuning_units, _ = get_tuning_eval_pools(seed=42)
53:         self.tuning_units = tuning_units
54:         self.residuals_by_bin = self._build_risk_table(oof_csv_path, self.tuning_units)
55: 
56:     def _build_risk_table(
57:         self, oof_csv_path: Union[str, Path], tuning_units: np.ndarray
58:     ) -> Dict[int, np.ndarray]:
59:         """
60:         Construct the empirical residual distribution table from Tuning pool engines only.
61:         Residual e = true_rul_raw - pred_rul.
62:         Bins are of width 10: [0, 10), [10, 20), ..., [120, 130).
63:         """
64:         df = pd.read_csv(oof_csv_path)
65:         df_tune = df[df["unit_number"].isin(tuning_units)].copy()
66:         df_tune["residual"] = df_tune["true_rul_raw"] - df_tune["pred_rul"]
67:         df_tune["bin"] = np.clip((df_tune["pred_rul"] // 10).astype(int), 0, 12)
68: 
69:         table: Dict[int, np.ndarray] = {}
70:         for b in range(13):
71:             sub = df_tune[df_tune["bin"] == b]["residual"].values
72:             if len(sub) == 0:
73:                 sub = np.array([0.0])
74:             table[b] = sub
75:         return table
76: 
77:     def get_bin(self, pred_rul: float) -> int:
78:         return int(np.clip(pred_rul // 10, 0, 12))
79: 
80:     def evaluate_risk(
81:         self, pred_rul: float, candidate_starts: np.ndarray
82:     ) -> Tuple[np.ndarray, np.ndarray]:
83:         """
84:         Compute failure probability P(R <= s) and expected operating cycles E[min(s, R)]
85:         for candidate start offsets s (s >= 0 relative to now).
86:         """
87:         if len(candidate_starts) == 0:
88:             return np.array([]), np.array([])
89: 
90:         if not self.use_intervals:
91:             # Agent-Point: deterministic R = pred_rul, no residual spread
92:             prob_fail = np.where(candidate_starts >= pred_rul, 1.0, 0.0)
93:             exp_min_r = np.minimum(candidate_starts, pred_rul).astype(float)
94:             return prob_fail, exp_min_r
95: 
96:         # Agent-Interval: empirical residual distribution
97:         b = self.get_bin(pred_rul)
98:         residuals = self.residuals_by_bin[b]  # shape (K,)
99:         # R = max(0.0, pred_rul + e)
100:         R = np.maximum(0.0, pred_rul + residuals)[:, np.newaxis]  # shape (K, 1)
101:         s = candidate_starts[np.newaxis, :]  # shape (1, M)
102: 
103:         prob_fail = np.mean(R <= s, axis=0)  # shape (M,)
104:         exp_min_r = np.mean(np.minimum(s, R), axis=0)  # shape (M,)
105:         return prob_fail, exp_min_r
106: 
107: 
108: class ResourceAgent:
109:     """
110:     Agent 2: Resource Agent.
111:     Manages spare parts inventory logistics and technician roster calendars.
112:     Ensures reservation consistency: reserved technician-cycles never exceed roster capacity.
113:     """
114: 
115:     def __init__(
116:         self,
117:         spare_lead_time: int = 8,
118:         preventive_duration: int = 2,
119:         horizon_W: int = 60,
120:     ) -> None:
121:         self.spare_lead_time = spare_lead_time
122:         self.preventive_duration = preventive_duration
123:         self.horizon_W = horizon_W
124:         self.capacity_roster: np.ndarray = np.array([], dtype=int)
125:         self.reserved_calendar: np.ndarray = np.array([], dtype=int)
126: 
127:     def reset(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
128:         """Initialize technician capacity roster and clear reservation calendar."""
129:         cal_len = max(len(capacity_roster), total_horizon + 300)
130:         pad_len = cal_len - len(capacity_roster)
131:         if pad_len > 0:
132:             pad_val = int(capacity_roster[-1]) if len(capacity_roster) > 0 else 4
133:             self.capacity_roster = np.pad(capacity_roster, (0, pad_len), constant_values=pad_val)
134:         else:
135:             self.capacity_roster = capacity_roster.copy()
136:         self.reserved_calendar = np.zeros(len(self.capacity_roster) + 100, dtype=int)
137: 
138:     def is_window_available(self, start_cycle: int, duration: int) -> bool:
139:         """Return True if capacity - reserved >= 1 on all duration consecutive cycles."""
140:         for c in range(start_cycle, start_cycle + duration):
141:             if c >= len(self.capacity_roster):
142:                 return False
143:             if (self.capacity_roster[c] - self.reserved_calendar[c]) < 1:
144:                 return False
145:         return True
146: 
147:     def filter_candidates(
148:         self,
149:         obs: EngineObservation,
150:         candidate_s: np.ndarray,
Created At: 2026-11-18T16:40:34+05:30
Completed At: 2026-11-18T16:40:34+05:30
Created At: 2026-11-18T16:40:34+05:30
Completed At: 2026-11-18T16:40:34+05:30
Showing lines 151-320 of 606
151:     ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
152:         """
153:         Evaluate feasibility of candidate start offsets s relative to current cycle (now).
154:         A candidate is feasible if:
155:         1. Spare part can arrive in time: (now + s) >= part_arrival
156:         2. Technician capacity is available for full duration D.
157:         """
158:         now = obs.current_cycle
159:         L = self.spare_lead_time
160:         D = self.preventive_duration
161: 
162:         if obs.order_already_placed and obs.part_arrival_cycle is not None:
163:             part_arrival = obs.part_arrival_cycle
164:         else:
165:             part_arrival = now + L
166: 
167:         feasible_mask = np.zeros(len(candidate_s), dtype=bool)
168:         diagnostics: List[Dict[str, Any]] = []
169: 
170:         for idx, s in enumerate(candidate_s):
171:             target_cycle = now + s
172:             part_ready = bool(target_cycle >= part_arrival)
173:             tech_ready = bool(self.is_window_available(target_cycle, D))
174:             is_feasible = bool(part_ready and tech_ready)
175:             feasible_mask[idx] = is_feasible
176: 
177:             diagnostics.append({
178:                 "start_cycle": int(target_cycle),
179:                 "offset_s": int(s),
180:                 "feasible": is_feasible,
181:                 "part_ready": part_ready,
182:                 "part_arrival": int(part_arrival),
183:                 "tech_ready": tech_ready,
184:             })
185: 
186:         return feasible_mask, diagnostics
187: 
188:     def reserve(self, start_cycle: int, duration: int) -> bool:
189:         """Reserve 1 technician for duration consecutive cycles starting at start_cycle."""
190:         if start_cycle + duration > len(self.capacity_roster):
191:             pad_val = int(self.capacity_roster[-1]) if len(self.capacity_roster) > 0 else 4
192:             pad_len = (start_cycle + duration + 100) - len(self.capacity_roster)
193:             self.capacity_roster = np.pad(self.capacity_roster, (0, pad_len), constant_values=pad_val)
194:             self.reserved_calendar = np.pad(self.reserved_calendar, (0, pad_len), constant_values=0)
195: 
196:         # Invariant check
197:         for c in range(start_cycle, start_cycle + duration):
198:             if (self.capacity_roster[c] - self.reserved_calendar[c]) < 1:
199:                 return False
200: 
201:         self.reserved_calendar[start_cycle : start_cycle + duration] += 1
202:         return True
203: 
204:     def release(self, start_cycle: int, duration: int) -> None:
205:         """Release a previously reserved technician window."""
206:         end_cycle = min(len(self.reserved_calendar), start_cycle + duration)
207:         if start_cycle < len(self.reserved_calendar):
208:             self.reserved_calendar[start_cycle : end_cycle] = np.maximum(
209:                 0, self.reserved_calendar[start_cycle : end_cycle] - 1
210:             )
211: 
212:     def find_earliest_available(self, earliest_cycle: int, duration: int) -> int:
213:         """Find earliest cycle >= earliest_cycle with sufficient technician capacity."""
214:         c = earliest_cycle
215:         max_search = len(self.capacity_roster) - duration
216:         while c <= max_search:
217:             if self.is_window_available(c, duration):
218:                 return c
219:             c += 1
220:         return earliest_cycle
221: 
222: 
223: class CostAgent:
224:     """
225:     Agent 3: Cost Agent.
226:     Evaluates expected cost per operating cycle C(s)/E(s) for candidate start cycles.
227:     The fleet pays for renewals; comparing C(s) alone would always pick the earliest window.
228:     The rate C(s)/E(s) properly accounts for wasted engine useful life.
229:     """
230: 
231:     def __init__(
232:         self,
233:         cost_preventive: float = 2000.0,
234:         cost_corrective: float = 10000.0,
235:         cost_production_loss: float = 100.0,
236:         cost_spare: float = 300.0,
237:         preventive_duration: int = 2,
238:         corrective_duration: int = 6,
239:     ) -> None:
240:         self.cost_preventive = cost_preventive
241:         self.cost_corrective = cost_corrective
242:         self.cost_production_loss = cost_production_loss
243:         self.cost_spare = cost_spare
244:         self.preventive_duration = preventive_duration
245:         self.corrective_duration = corrective_duration
246: 
247:         self.c_pm_total = self.cost_preventive + self.cost_spare + self.cost_production_loss * self.preventive_duration
248:         self.c_fail_total = self.cost_corrective + self.cost_spare + self.cost_production_loss * self.corrective_duration
249: 
250:     def evaluate_cost_rates(
251:         self,
252:         age: int,
253:         candidate_s: np.ndarray,
254:         prob_fail: np.ndarray,
255:         exp_min_r: np.ndarray,
256:     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
257:         """
258:         Compute C(s), E(s), and C(s)/E(s) for candidate start offsets.
259:         """
260:         c_s = (1.0 - prob_fail) * self.c_pm_total + prob_fail * self.c_fail_total
261:         e_s = float(age) + exp_min_r
262:         denom = np.maximum(e_s, 1e-6)
263:         cost_rates = c_s / denom
264:         return c_s, e_s, cost_rates
265: 
266: 
267: class CoordinatorAgent:
268:     """
269:     Agent 4: Scheduling / Coordinator Agent.
270:     Coordinates fleet-level planning pass, orchestrates Prognostic, Resource, and Cost agents,
271:     manages priority queues, escalation procedures, and decision logging to JSONL.
272:     """
273: 
274:     def __init__(
275:         self,
276:         prognostic: PrognosticAgent,
277:         resource: ResourceAgent,
278:         cost: CostAgent,
279:         replan_k: int = 5,
280:         horizon_W: int = 60,
281:         spare_lead_time: int = 8,
282:         decision_log_path: Optional[Union[str, Path]] = None,
283:     ) -> None:
284:         self.prognostic = prognostic
285:         self.resource = resource
286:         self.cost = cost
287:         self.replan_k = replan_k
288:         self.horizon_W = horizon_W
289:         self.spare_lead_time = spare_lead_time
290:         self.decision_log_path = Path(decision_log_path) if decision_log_path else None
291: 
292:         # Slot tracking: slot_id -> dict with current plan
293:         self.slot_plans: Dict[int, Dict[str, Any]] = {}
294:         self.decision_records: List[Dict[str, Any]] = []
295: 
296:     def reset(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
297:         self.resource.reset(total_horizon, capacity_roster)
298:         self.slot_plans.clear()
299:         self.decision_records.clear()
300:         if self.decision_log_path and self.decision_log_path.exists():
301:             try:
302:                 self.decision_log_path.unlink()
303:             except Exception:
304:                 pass
305: 
306:     def on_renewal(self, slot_id: int) -> None:
307:         """Callback when an engine renewal occurs; clear reservation and plan for slot."""
308:         if slot_id in self.slot_plans:
309:             plan = self.slot_plans[slot_id]
310:             if plan.get("reserved", False) and "planned_start" in plan:
311:                 self.resource.release(plan["planned_start"], self.resource.preventive_duration)
312:             del self.slot_plans[slot_id]
313: 
314:     def on_engine_failure(self, slot_id: int) -> None:
315:         """Callback if an engine fails before planned start; release reserved tech window."""
316:         if slot_id in self.slot_plans:
317:             plan = self.slot_plans[slot_id]
318:             if plan.get("reserved", False) and "planned_start" in plan:
319:                 self.resource.release(plan["planned_start"], self.resource.preventive_duration)
320:                 plan["reserved"] = False
Created At: 2026-11-18T16:40:39+05:30
Completed At: 2026-11-18T16:40:39+05:30
Created At: 2026-11-18T16:40:39+05:30
Completed At: 2026-11-18T16:40:39+05:30
Showing lines 320-450 of 606
320:                 plan["reserved"] = False
321: 
322:     def plan_fleet(
323:         self,
324:         current_cycle: int,
325:         observations: List[EngineObservation],
326:     ) -> Dict[int, Dict[str, Any]]:
327:         """
328:         Execute fleet-wide planning pass at current_cycle.
329:         Returns dictionary of updated action orders per slot.
330:         """
331:         now = current_cycle
332:         L = self.spare_lead_time
333:         W = self.horizon_W
334:         D = self.resource.preventive_duration
335: 
336:         # Step 1: Identify engines within planning horizon (predicted RUL <= L + W)
337:         candidate_engines: List[EngineObservation] = []
338:         for obs in observations:
339:             if obs.pred_rul <= (L + W):
340:                 candidate_engines.append(obs)
341: 
342:         # Step 2: Filter engines requiring planning or re-planning
343:         to_plan: List[EngineObservation] = []
344:         for obs in candidate_engines:
345:             s_id = obs.slot_id
346:             if s_id not in self.slot_plans:
347:                 to_plan.append(obs)
348:             else:
349:                 plan = self.slot_plans[s_id]
350:                 # Re-plan every k cycles
351:                 is_periodic = (now % self.replan_k == 1)
352:                 # Re-plan immediately if lower bound falls below planned start
353:                 planned_start = plan.get("planned_start", now + 999)
354:                 lower_dropped = (obs.lower_rul < (planned_start - now))
355:                 if is_periodic or lower_dropped:
356:                     to_plan.append(obs)
357: 
358:         if not to_plan:
359:             return {}
360: 
361:         # Step 3: Sort by urgency (lowest lower bound first)
362:         to_plan.sort(key=lambda o: o.lower_rul)
363: 
364:         action_orders: Dict[int, Dict[str, Any]] = {}
365: 
366:         # Step 4: Plan each engine in urgency order
367:         for obs in to_plan:
368:             s_id = obs.slot_id
369:             # If engine already had a reservation, temporarily release it during re-plan
370:             had_prior_plan = s_id in self.slot_plans
371:             if had_prior_plan:
372:                 prior_plan = self.slot_plans[s_id]
373:                 if prior_plan.get("reserved", False) and "planned_start" in prior_plan:
374:                     self.resource.release(prior_plan["planned_start"], D)
375:                     prior_plan["reserved"] = False
376: 
377:             # Candidate search range:
378:             # Agent-Point: s < pred_rul (hard limit)
379:             # Agent-Interval: s <= pred_rul (hard limit)
380:             if not self.prognostic.use_intervals:
381:                 max_s = max(0, int(np.floor(obs.pred_rul - 1.0)))
382:             else:
383:                 max_s = max(0, int(np.floor(obs.pred_rul)))
384: 
385:             candidate_s = np.arange(0, max_s + 1, dtype=int)
386: 
387:             # Drop cycles where spare cannot arrive or no technician capacity
388:             feasible_mask, diag_list = self.resource.filter_candidates(obs, candidate_s)
389:             feasible_s = candidate_s[feasible_mask]
390: 
391:             chosen_cycle: Optional[int] = None
392:             escalated = False
393:             best_cost_rate = 999999.0
394:             order_spare_now = not obs.order_already_placed
395: 
396:             if len(feasible_s) > 0:
397:                 # Ask Cost Agent for cost rates of remaining cycles
398:                 prob_fail, exp_min_r = self.prognostic.evaluate_risk(obs.pred_rul, feasible_s)
399:                 c_s, e_s, cost_rates = self.cost.evaluate_cost_rates(obs.age, feasible_s, prob_fail, exp_min_r)
400: 
401:                 best_idx = int(np.argmin(cost_rates))
402:                 best_s = int(feasible_s[best_idx])
403:                 best_cost_rate = float(cost_rates[best_idx])
404:                 chosen_cycle = now + best_s
405: 
406:                 # Reserve technician capacity
407:                 reserved = self.resource.reserve(chosen_cycle, D)
408: 
409:                 # Attach cost rates to diagnostics for JSONL logging
410:                 cost_idx = 0
411:                 for d in diag_list:
412:                     if d["feasible"]:
413:                         d["cost_rate"] = float(cost_rates[cost_idx])
414:                         d["prob_fail"] = float(prob_fail[cost_idx])
415:                         cost_idx += 1
416:                     else:
417:                         d["cost_rate"] = None
418:                         d["prob_fail"] = None
419: 
420:                 reason = (
421:                     f"Selected cycle {chosen_cycle} (offset +{best_s} cycles) with lowest expected cost rate "
422:                     f"${best_cost_rate:.2f}/cycle (prob_fail={prob_fail[best_idx]*100:.1f}%, age={obs.age})."
423:                 )
424:             else:
425:                 # ESCALATION: No feasible window in candidate set
426:                 escalated = True
427:                 order_spare_now = not obs.order_already_placed
428:                 earliest_part = obs.part_arrival_cycle if obs.order_already_placed and obs.part_arrival_cycle else (now + L)
429:                 chosen_cycle = self.resource.find_earliest_available(earliest_part, D)
430:                 reserved = self.resource.reserve(chosen_cycle, D)
431: 
432:                 for d in diag_list:
433:                     d["cost_rate"] = None
434:                     d["prob_fail"] = None
435: 
436:                 reason = (
437:                     f"ESCALATION: No feasible maintenance window within predicted RUL {obs.pred_rul:.1f}. "
438:                     f"Secured earliest feasible window at cycle {chosen_cycle} (part arrives cycle {earliest_part})."
439:                 )
440: 
441:             # Store plan
442:             plan_record = {
443:                 "slot_id": s_id,
444:                 "unit_number": obs.unit_number,
445:                 "planned_start": chosen_cycle,
446:                 "order_cycle": now if order_spare_now else self.slot_plans.get(s_id, {}).get("order_cycle", now),
447:                 "part_arrival": now + L if order_spare_now else obs.part_arrival_cycle,
448:                 "escalated": escalated,
449:                 "reserved": reserved,
450:                 "cost_rate": best_cost_rate,
Created At: 2026-11-18T16:40:43+05:30
Completed At: 2026-11-18T16:40:43+05:30
Created At: 2026-11-18T16:40:43+05:30
Completed At: 2026-11-18T16:40:43+05:30
Showing lines 451-520 of 606
451:                 "urgency_lower": obs.lower_rul,
452:             }
453:             self.slot_plans[s_id] = plan_record
454: 
455:             # Decision JSONL record
456:             decision_entry = {
457:                 "engine": f"E-{obs.unit_number:03d}",
458:                 "cycle": now,
459:                 "predicted_rul": float(obs.pred_rul),
460:                 "interval": [float(obs.lower_rul), float(obs.upper_rul)],
461:                 "candidates": diag_list,
462:                 "chosen_cycle": int(chosen_cycle),
463:                 "escalated": bool(escalated),
464:                 "spare_order_cycle": int(plan_record["order_cycle"]),
465:                 "reason": reason,
466:             }
467:             self.decision_records.append(decision_entry)
468:             if self.decision_log_path:
469:                 with open(self.decision_log_path, "a", encoding="utf-8") as f:
470:                     f.write(json.dumps(decision_entry) + "\n")
471: 
472:             action_orders[s_id] = {
473:                 "order_spare": order_spare_now,
474:                 "planned_start": chosen_cycle,
475:                 "priority": float(obs.lower_rul),
476:                 "escalated": escalated,
477:             }
478: 
479:         return action_orders
480: 
481: 
482: class AgenticPolicy(BasePolicy):
483:     """
484:     Agentic Maintenance Decision Policy for Turbofan Fleet Simulation.
485:     Wraps the 4 collaborative agents (Prognostic, Resource, Cost, Coordinator)
486:     into the FleetSimulator policy interface.
487:     """
488: 
489:     def __init__(
490:         self,
491:         use_intervals: bool = True,
492:         replan_k: int = 5,
493:         horizon_W: int = 60,
494:         spare_lead_time: int = 8,
495:         oof_csv_path: Union[str, Path] = DEFAULT_OOF_PATH,
496:         config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
497:         decision_log_path: Optional[Union[str, Path]] = DEFAULT_DECISION_LOG_PATH,
498:     ) -> None:
499:         name = "Agent-Interval" if use_intervals else "Agent-Point"
500:         super().__init__(name=name)
501:         self.use_intervals = use_intervals
502:         self.replan_k = replan_k
503:         self.horizon_W = horizon_W
504:         self.spare_lead_time = spare_lead_time
505: 
506:         # Load parameters from YAML
507:         with open(config_path, "r", encoding="utf-8") as f:
508:             data = yaml.safe_load(f)
509:         costs = data.get("costs", {})
510:         ops = data.get("operations", {})
511: 
512:         c_pm = float(costs.get("preventive_maintenance", 2000.0))
513:         c_fail = float(costs.get("corrective_failure", 10000.0))
514:         c_loss = float(costs.get("production_loss_per_cycle", 100.0))
515:         c_part = float(costs.get("spare_part", 300.0))
516:         d_pm = int(ops.get("preventive_duration", 2))
517:         d_fail = int(ops.get("corrective_duration", 6))
518: 
519:         # Instantiate sub-agents
520:         self.prognostic = PrognosticAgent(oof_csv_path=oof_csv_path, use_intervals=use_intervals)
Created At: 2026-11-18T16:40:47+05:30
Completed At: 2026-11-18T16:40:48+05:30
Created At: 2026-11-18T16:40:47+05:30
Completed At: 2026-11-18T16:40:48+05:30
Showing lines 521-606 of 606
521:         self.resource = ResourceAgent(spare_lead_time=spare_lead_time, preventive_duration=d_pm, horizon_W=horizon_W)
522:         self.cost = CostAgent(
523:             cost_preventive=c_pm,
524:             cost_corrective=c_fail,
525:             cost_production_loss=c_loss,
526:             cost_spare=c_part,
527:             preventive_duration=d_pm,
528:             corrective_duration=d_fail,
529:         )
530:         self.coordinator = CoordinatorAgent(
531:             prognostic=self.prognostic,
532:             resource=self.resource,
533:             cost=self.cost,
534:             replan_k=replan_k,
535:             horizon_W=horizon_W,
536:             spare_lead_time=spare_lead_time,
537:             decision_log_path=decision_log_path,
538:         )
539: 
540:         self.current_action_orders: Dict[int, Dict[str, Any]] = {}
541:         self.num_escalations = 0
542: 
543:     def reset(self) -> None:
544:         self.current_action_orders.clear()
545:         self.num_escalations = 0
546: 
547:     def init_simulation(self, total_horizon: int, capacity_roster: np.ndarray) -> None:
548:         """Called by simulator before cycle loop to initialize roster calendars."""
549:         self.reset()
550:         self.coordinator.reset(total_horizon, capacity_roster)
551: 
552:     def on_renewal(self, slot_id: int) -> None:
553:         self.coordinator.on_renewal(slot_id)
554:         if slot_id in self.current_action_orders:
555:             del self.current_action_orders[slot_id]
556: 
557:     def on_engine_failure(self, slot_id: int) -> None:
558:         self.coordinator.on_engine_failure(slot_id)
559: 
560:     def before_cycle(
561:         self,
562:         current_cycle: int,
563:         observations: List[EngineObservation],
564:     ) -> None:
565:         """Execute fleet planning pass and record decisions."""
566:         orders = self.coordinator.plan_fleet(current_cycle, observations)
567:         for s_id, ord_dict in orders.items():
568:             if ord_dict.get("escalated", False):
569:                 self.num_escalations += 1
570:             self.current_action_orders[s_id] = ord_dict
571: 
572:     def decide(
573:         self,
574:         slot_id: int,
575:         current_cycle: int,
576:         operating_cycles: int,
577:         pred_rul: float,
578:         lower_rul: float,
579:         upper_rul: float,
580:         true_rul: float,
581:         unit_number: int,
582:         engine_cycle: int,
583:         order_already_placed: bool,
584:         fleet_context: Optional[Dict[str, Any]] = None,
585:     ) -> bool:
586:         """
587:         Check if an order was placed by the Coordinator for this slot at this cycle.
588:         """
589:         if order_already_placed:
590:             return False
591: 
592:         if slot_id in self.current_action_orders:
593:             ord_dict = self.current_action_orders[slot_id]
594:             if ord_dict.get("order_spare", False):
595:                 return True
596:         return False
597: 
598:     def get_earliest_start(self, slot_id: int, default: int) -> int:
599:         if slot_id in self.coordinator.slot_plans:
600:             return int(self.coordinator.slot_plans[slot_id].get("planned_start", default))
601:         return default
602: 
603:     def get_priority(self, slot_id: int, default: float = 0.0) -> float:
604:         if slot_id in self.coordinator.slot_plans:
605:             return float(self.coordinator.slot_plans[slot_id].get("urgency_lower", default))
606:         return default