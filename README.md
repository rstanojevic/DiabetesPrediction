# Predikcija dijabetesa primenom više algoritama mašinskog učenja

Diplomski rad — uporedna analiza šest algoritama klasifikacije na
Pima Indians Diabetes skupu podataka.

## Skup podataka

Pima Indians Diabetes Database — 768 pacijentkinja, 8 atributa,
binarni ishod (268 pozitivnih, 500 negativnih).

Originalni izvor: Smith, J.W., Everhart, J.E., Dickson, W.C.,
Knowler, W.C., Johannes, R.S. (1988). *Using the ADAP learning
algorithm to forecast the onset of diabetes mellitus*. Podaci
potiču iz National Institute of Diabetes and Digestive and Kidney
Diseases. Preuzeto sa javno dostupne kopije na GitHub-u.

Fajl se nalazi u `data/raw/diabetes.csv`.

## Modeli

Logistic Regression, KNN, SVM, Decision Tree, Random Forest, XGBoost.

Svaki model je kombinovan sa četiri imputacione strategije
(mean, median, KNN, MICE), što daje 24 eksperimenta.

## Struktura projekta

diabetes-prediction/
├── data/
│   ├── raw/diabetes.csv
│   └── processed/              # opciono, nakon zamene nula
├── src/
│   ├── __init__.py
│   ├── config.py               # putanje, RANDOM_STATE, CV parametri
│   ├── data_loader.py          # učitavanje, nule -> NaN
│   ├── preprocessing.py        # imputeri, skaleri, gradnja Pipeline-a
│   ├── models.py               # 6 modela + mreže hiperparametara
│   ├── evaluation.py           # metrike, matrice, krive
│   ├── statistics.py           # Wilcoxon, Friedman, Nemenyi
│   └── explainability.py       # SHAP
├── scripts/
│   ├── run_eda.py              # -> figures/eda/
│   ├── run_experiments.py      # -> tables/cv_results.csv, models/*.pkl
│   ├── run_evaluation.py       # -> figures/, tables/
│   └── run_shap.py             # -> figures/shap/
├── reports/
│   ├── figures/                # PDF/SVG za rad
│   ├── tables/                 # CSV -> tabele u radu
│   └── logs/                   # tekstualni ispisi eksperimenata
├── models/                     # joblib .pkl
├── requirements.txt
└── README.md

## Pokretanje skripti

Skripte se pokreću iz korena projekta, ovim redom:
python -m scripts.run_eda # ~5 s
python -m scripts.run_experiments # ~35 min
python -m scripts.run_evaluation # ~10 s
python -m scripts.run_shap # ~5 s


`run_experiments.py` mora da se pokrene pre ostale dve, jer one
koriste `reports/tables/cv_results.csv` i obučene modele iz
`models/`.

Folder `models/` nije u repozitorijumu (24 fajla(.pkl), nekoliko MB),
ali su svi rezultati u `reports/` commitovani, pa se tabele i
slike mogu pregledati bez pokretanja eksperimenta.

## Metodologija

**Nedostajuće vrednosti.** U kolonama Glucose, BloodPressure,
SkinThickness, Insulin i BMI nula je fiziološki nemoguća i
tretirana je kao nedostajuća vrednost. U koloni Pregnancies nula
je validna i zadržana je.

**Podela podataka.** Test skup od 20% je izdvojen stratifikovano i
nije korišćen ni u jednoj fazi osim konačne evaluacije. Nad
preostalih 80% primenjena je ponovljena stratifikovana k-fold
kros-validacija (10 foldova × 5 ponavljanja = 50 procena).

Validacioni skup time nije fiksan nego rotirajući. Razlog je
veličina skupa: pri n = 768 jedan validacioni skup sadržao bi oko
120 uzoraka, a izmerena varijabilnost po foldovima (ROC-AUC od
0.655 do 0.934) pokazuje da bi takva procena bila nepouzdana za
izbor hiperparametara.

**Curenje podataka.** Imputacija i skaliranje su koraci unutar
`Pipeline`-a, pa se uče isključivo na trening delu svakog folda.

**Reproducibilnost.** Svi izvori nasumičnosti su fiksirani
(`RANDOM_STATE = 42`). Celokupan eksperiment je pokrenut dvaput i
sve 1200 pojedinačnih ocena po foldu su identične.

## Rezultati

Kros-validacija (ROC-AUC, prosek ± std preko 50 foldova):

| Model | ROC-AUC |
|---|---|
| SVM | 0.8476 ± 0.0432 |
| XGBoost | 0.8447 ± 0.0479 |
| Logistic Regression | 0.8441 ± 0.0435 |
| Random Forest | 0.8407 ± 0.0482 |
| KNN | 0.8382 ± 0.0467 |
| Decision Tree | 0.8025 ± 0.0530 |

Test skup (n = 154): XGBoost 0.8207, SVM 0.8172, Logistic
Regression 0.8104, Random Forest 0.8091, KNN 0.8046,
Decision Tree 0.7936.

Friedmanov test pokazuje značajnu razliku među modelima
(χ² = 48.58, p < 0.001). Naknadno poređenje Wilcoxonovim testom uz
Holmovu korekciju pokazuje da je razlika u potpunosti posledica
slabijih performansi stabla odlučivanja: značajno je 5 od 15
parova, a svih pet uključuje stablo. Nijedna razlika među
preostalih pet modela nije statistički značajna (p > 0.08).

Bazna linija (predviđanje većinske klase) iznosi 0.649 accuracy.