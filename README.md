# Aircraft component removal risk

This proof of concept scores each installed component for an unscheduled removal within its next 30 flight cycles. All aircraft, serials, sensor measurements and removals are synthetic.

[Assessment report](REPORT.md) · [Data dictionary](data/README.md) · [Model comparison](results/comparison.csv) · [Example cases](results/examples.csv) · [Latest fleet scores](results/latest_fleet_scores.csv)

## Reproduce the assessment

Use 64-bit Python 3.13. From this directory in PowerShell, run the following commands.

~~~powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python simulate.py
.venv\Scripts\python -m unittest -v
.venv\Scripts\python assess.py
~~~

On macOS or Linux, use the Python executable in `.venv/bin`. The fixed-seed mock data is included, so the simulation step is optional. The assessment verifies the published labels against the removal ledger before fitting. It exports the validation threshold sweep, held-out scores, policy decisions and report. To score another compatible file of active-component histories using the fitted winner, run the command below after training.

~~~powershell
.venv\Scripts\python score.py data/observations.csv.gz results/latest_fleet_scores.csv
~~~

The scoring input needs enough preceding observations for each installation to build trailing features. It may omit the outcome label. Output covers every active component in the last fleet snapshot. The daily ticket rules are evaluated separately in the historical replay.
