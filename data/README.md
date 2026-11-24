# Synthetic fleet data

[Daily active-component snapshots](observations.csv.gz) and the [removal ledger](removals.csv) are the public mock dataset. They were generated together by the seeded simulator. An observation is taken after that day's flights, maintenance and replacements. The fleet has eight instrumented positions per aircraft. Every position has exactly one active installation at each scoring date.

| Observation field | Meaning |
|---|---|
| date | End-of-day scoring date, ISO calendar format |
| aircraft_id, airframe_family | Synthetic airframe identifier and family |
| position | Position on the aircraft |
| serial, install_id | Physical component identifier and installation episode key. Each mock serial is installed once. |
| part_family | Synthetic rotating component type |
| aircraft_cycle | Cumulative number of cycles flown by the aircraft at scoring |
| cycles_today | Flights completed by the aircraft on this date |
| cycles_since_install, cycles_since_service | Age of this installation and elapsed cycles since its latest recorded service |
| vibration, temperature_c | Current synthetic instrument readings in mm/s and degrees Celsius. Blank readings and injected out-of-range values model data quality issues. |
| fault_code | NONE, VIB or THERM. A blank value means the report is unavailable. |
| label_30fc | 1 if this installation exits unscheduled within the next 1–30 flown cycles, 0 for a known negative, blank for unresolved follow-up |

The removal ledger records the date, aircraft, position, serial and installation ID of each exit, along with the aircraft's removal cycle and the reason (planned or unscheduled). Each installation has at most one exit. A planned exit before the horizon closes the risk window for that installation. Freshly installed components are first observed after the old installation's exit, at the same aircraft cycle.

Positive records overlap when successive observations predict the same event. Count unique unscheduled ledger entries for event recall. Model inputs are explicitly selected in the feature module. The ledger and label columns are used for fitting targets and evaluation only. The compressed observations can be read directly with pandas.read_csv.
