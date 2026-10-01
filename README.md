# QKD Eavesdropping Research Code

`main.ipynb` retains the original value-targeted experiment and appends the new availability-attack direction in Cells 30 onward. The new simulator and PPO helpers are in `threat_aware_qkd.py` and `threat_aware_rl.py`.

- `main_value_targeted_legacy.ipynb` preserves the earlier edited draft.
- `prototypes/threat_aware_initial/` preserves the initial standalone prototype.
- Generated models, result tables, and figures are excluded from Git.

Install `requirements.txt` and run the notebook in order. Cell 30g guards the PPO sweep; enable it after reviewing the HMM results. The full grid and training configurations are provided but can take substantial runtime.
