# Observability: run_id, logs, failed events

- **run_id** (`20261019T051400Z-a1b2c3`): one per publisher / consumer process. Publisher messages carry it
  (`source_run_id` in the DB); the consumer's own id is `run_id`.
- **Logs**: console = readable lines; `results/logs/<component>_<run_id>.jsonl` = JSON lines.
- **`runs` table**: start/end, heartbeat, status (`running`, `stopped`, `crashed`, `unclean_shutdown`), config, final counters.
- **`failed_events` table**: every rejected event, with reason and the raw payload (truncated to 1000 chars).

| reason | meaning |
|---|---|
| `malformed` | bad JSON, missing field, wrong type |
| `non_finite_value` | NaN / inf in a reading |
| `out_of_order` | seq not newer than the last one seen for that device (duplicate or late) |
| `gap_reset` | device silent longer than `max_gap_s`; its window was restarted |
| `scoring_error` | the model raised on a batch |

Counted but NOT stored per event (see `runs.summary`): `warmup` (window not full), `dropped` (queue full),
`db_error` (write failed).

```sql
-- failures by reason, last hour
SELECT reason, COUNT(*) FROM failed_events WHERE ts > strftime('%s','now') - 3600 GROUP BY reason;
-- runs and how they ended
SELECT run_id, component, status, datetime(started_at,'unixepoch') FROM runs ORDER BY started_at DESC;
-- everything one consumer run did
SELECT reason, COUNT(*) FROM failed_events WHERE run_id = '<run_id>' GROUP BY reason;
-- compare models
SELECT model, COUNT(*), AVG(is_anomaly), AVG(e2e_ms) FROM readings GROUP BY model;
```

Sliding window: per device, last W readings; features = current z, rolling mean, rolling std, delta.
Warm-up = W readings. A stuck sensor shows up as `*_std ~ 0`, which a single-reading model cannot see.

**Caveat (measured in the simulator):** an Isolation Forest still flags stuck sensors poorly even with windows,
because the signal is 1 of 24 features (recall ~5-10% at a 1% false-positive budget), while spikes improve
(~0.67 -> ~0.85). An explicit rule (`any rolling std < 0.02`) caught ~78% of stuck readings with ~0 false positives;
the rest are the first W readings of each fault, before the window fills. Use the model for spikes and a rule for flatlines.
Also note: after a spike, windowed alerts keep firing until it leaves the window, so "false positives" are inflated.
Windowed models are served by the consumer only; the stateless API returns 501 for them.

## Flatline rule

Stuck sensors are caught by a rule, not by the model: `scripts/consume_score.py --flat-eps 0.02`
flags any sensor whose rolling std (z-units) is below the threshold (`0` disables it). The reading's
`alert_reason` column says why it was flagged: `model`, `flatline:<sensors>`, or `model+flatline:<sensors>`.

- Only windowed models get the rule (it needs the rolling std).
- Detection starts once the window is full, so the first W readings of a fault are missed.
- One alert per reading while a sensor stays stuck (no de-duplication yet).
- A sensor that is legitimately constant will trip it; raise `--flat-eps` carefully or exclude it.

```sql
SELECT alert_reason, COUNT(*) FROM readings WHERE is_anomaly = 1 GROUP BY alert_reason;
```
