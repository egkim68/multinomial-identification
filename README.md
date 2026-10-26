# Which Zeros? Identification Choices and Sparse Reporting in Multinomial Logistic Regression

Code and protocol for the simulation study of the same title (Eungi Kim, Keimyung University).
Requires Python 3 with numpy, scipy, pandas, and statsmodels (scikit-learn only for `solver_check.py`).

Run from `code/` (output folders are created automatically):

```
mkdir -p output && python3 theory_checks.py
python3 solver_check.py
python3 run_resumable.py main 8823 50  output/main
python3 run_resumable.py wald 8824 100 output/wald
python3 run_resumable.py sens 9901 50  output/sens
python3 addB_resumable.py b1 9911 100  output/addB
python3 addB_resumable.py b2 9912 25   output/addB
python3 analysis.py output/main/multinom_results.csv output/wald/wald_results.csv output/analysis
python3 analysis_addenda.py output/sens/sensitivity_results.csv output/addB/addB_b1.csv output/addB/addB_b2.csv output/addenda
SIM_SEED=9921 python3 addC_sim.py c2 50 output/addC
SIM_SEED=9931 python3 addC_sim.py c1 25 output/addC
python3 analysis_addC.py output/addC/addC_c1.csv output/addC/addC_c2.csv output/addC/summary
```

The seeds above are those fixed in `PROTOCOL.md`. Propositions 1-2 in the protocol are Results 1-2 of the paper. For the survey illustration, export the data in R with
`write.csv(carData::Chile, "Chile.csv")` and run `python3 illustration_chile.py Chile.csv output/chile`.

License: MIT.
