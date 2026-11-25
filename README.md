# Choosing the reference category in multinomial logistic regression

Code for the paper "The Effect of the Reference Category on Wald Tests in Multinomial Logistic Regression" (Eungi Kim, Keimyung University). The design and hypotheses are in `PROTOCOL.md`.

Requires Python 3 with numpy, scipy, pandas, statsmodels and matplotlib. The reported results were produced with Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5 and statsmodels 0.15.0. Other versions should give the same results up to small numerical differences. Run the scripts from `code/`. Output folders are created automatically.

```
python3 reference_sim.py 20261125 100 output/reference
python3 analysis.py output/reference/reference_results.csv output/analysis
python3 frequency_check.py output/frequency
python3 chile.py Chile.csv output/chile
python3 make_figures.py output/frequency output/analysis output/figures
```

To get `Chile.csv`, export the survey data in R with `write.csv(carData::Chile, "Chile.csv")`. The figures are written as PNG files by `make_figures.py`, so they can be redrawn without rerunning the simulation.

License: MIT.
