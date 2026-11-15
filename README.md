# Independent Validation in Python: paper and demo

Manuscript source and demo code for

> von Oertzen, Thede, Diemerling, H., & von Oertzen, Timo (2026). How to use
> independent validation in Python. *Methodology, 22*(2), Article e18873.
> https://doi.org/10.5964/meth.18873

The method itself is available as a package:

```bash
pip install independent-validation
```

Package source: https://github.com/jonasthedevonoertzen/IV

## What Independent Validation does

Cross validation reuses the same samples for training and testing. The test
outcomes are therefore dependent, which inflates the variance of the accuracy
estimate and makes significance statements about classifiers unreliable.

Independent Validation (IV) tests every sample before it is used for training.
The test outcomes are independent, so the accuracy of a classifier can be
described by a full Bayesian posterior distribution. This allows statements
such as "the probability that classifier A is more accurate than classifier B"
or "the probability that the classifier is better than guessing".

## Minimal example

```python
from sklearn.datasets import load_wine
from sklearn.svm import SVC
import independent_validation as iv

X, y = load_wine(return_X_y=True)

iv_svm = iv.IV(X, y, SVC(gamma="scale"))
iv_svm.run_iv()               # test each sample, then add it to the training set
iv_svm.compute_posterior()    # MCMC estimate of the posterior
bacc = iv_svm.get_bacc_dist() # posterior of the balanced accuracy

print("MAP:", bacc.map())
print("95% interval:", bacc.ppf(0.025), bacc.ppf(0.975))
```

The printed values change from run to run, because `IV` shuffles the data and
the posterior is sampled by MCMC. Call `numpy.random.seed()` first for
repeatable output.

## Contents

| Path | Contents |
|---|---|
| `main.py` | Demo from the paper on the wine data set: group difference, comparison of four classifiers (balanced accuracy and accuracy), accuracy as a function of training set size |
| `plots/` | Figures 1 to 4 of the paper, produced by `main.py` |
| `demo/` | Additional example plots (synthetic data, Titanic); the code that produced them is not in this repository |
| `main.tex`, `apa7.cls`, `bibliography.bib` | Manuscript source in LaTeX (APA 7 class), pre-publication version |
| `iv_in_python.md`, `_extensions/` | Quarto version of the manuscript with the apaquarto extension; its front matter currently holds template placeholders |
| `fragments.txt` | Unused text fragments from drafting |
| `iv_in_python.pdf` | Early rendering of the draft (February 2025, title "Implementation of independent validation in Python"); for the final text see the published article |
| `.github/workflows/workflow.yml` | GitHub Actions job that renders `iv_in_python.md` to PDF with Quarto |
| `CITATION.cff` | Citation metadata |

## Reproducing the figures

```bash
pip install independent-validation scikit-learn matplotlib seaborn
python main.py
```

The figures are written to `plots/`.

## License

MIT
