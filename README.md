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
(mean, median, KNN, MICE), što daje 24 eksperimenta. Za svaki
eksperiment sprovedena je pretraga hiperparametara nad 50 foldova,
što ukupno iznosi oko 83.000 treniranja modela.

## Metrike evaluacije

Za svaki model i svaki fold računa se šest metrika:

| Metrika | Šta meri | Bazna linija |
|---|---|---|
| Accuracy | udeo tačnih predikcija | 0.651 (većinska klasa) |
| Precision | pouzdanost pozitivnog nalaza | 0.349 |
| Recall | udeo prepoznatih dijabetičarki | — |
| F1 | harmonijska sredina precision-a i recall-a | — |
| ROC-AUC | kvalitet rangiranja, nezavisno od praga | 0.5 |
| PR-AUC | rangiranje iz ugla pozitivne klase | 0.349 |

ROC-AUC je korišćen kao primarna metrika pri izboru hiperparametara
jer ne zavisi od praga odlučivanja i omogućava poređenje sa
rezultatima objavljenim u literaturi. Ostale metrike se izveštavaju
uporedo, jer pojedinačna metrika daje pristrasnu sliku — poredak
modela se menja u zavisnosti od toga koja se posmatra.

## Struktura projekta

```text
diabetes-prediction/
├── data/
│   └── raw/diabetes.csv
├── src/
│   ├── __init__.py
│   ├── config.py               # putanje, RANDOM_STATE, CV parametri, prostori pretrage
│   ├── data_loader.py          # učitavanje, nule -> NaN, podela na trening i test
│   ├── preprocessing.py        # imputeri, skaleri, gradnja Pipeline-a
│   ├── models.py               # 6 modela + mreže hiperparametara + build_pipeline
│   ├── evaluation.py           # izbor najbolje imputacione strategije po modelu
│   ├── statistics.py           # Friedman, Wilcoxon, Holmova korekcija
│   ├── explainability.py       # SHAP
│   └── plotting.py             # podešavanja matplotlib-a i snimanje figura
├── scripts/
│   ├── run_eda.py              # eksplorativna analiza
│   ├── run_experiments.py      # treniranje i pretraga hiperparametara
│   ├── run_evaluation.py       # metrike, grafikoni, statistički testovi
│   └── run_shap.py             # SHAP analiza najboljeg modela
├── reports/
│   ├── figures/                # 11 grafikona (PDF)
│   ├── tables/                 # 6 tabela (CSV)
│   └── logs/                   # log eksperimenta
├── models/                     # 24 obučena pipeline-a (joblib .pkl)
├── requirements.txt
└── README.md
```

## Pokretanje skripti

Projekat je razvijan i testiran na Python 3.13.7.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Skripte se pokreću iz korena projekta, ovim redom:

```bash
python -m scripts.run_eda           # ~5 s
python -m scripts.run_experiments   # ~35 min
python -m scripts.run_evaluation    # ~10 s
python -m scripts.run_shap          # ~5 s
```

`run_experiments.py` mora da se pokrene pre `run_evaluation.py` i
`run_shap.py`, jer one koriste `reports/tables/cv_results.csv` i
obučene modele iz `models/`.

Folder `models/` nije u repozitorijumu (24 fajla `.pkl`, nekoliko MB),
ali su svi rezultati u `reports/` commitovani, pa se tabele i
grafikoni mogu pregledati bez pokretanja eksperimenta.

## Metodologija

**Nedostajuće vrednosti.** U kolonama Glucose, BloodPressure,
SkinThickness, Insulin i BMI nula je fiziološki nemoguća i
tretirana je kao nedostajuća vrednost. U koloni Pregnancies nula
je validna i zadržana je. Insulinu nedostaje 48.7% vrednosti,
a kožnom naboru 29.6%.

**Mehanizam nedostajanja.** Poređenjem grupa sa izmerenim i
neizmerenim insulinom utvrđeno je da se one razlikuju po starosti
(35.8 naspram 30.8 godina), ali ne i po raspodeli klasa. Obrazac
odgovara MAR mehanizmu, što opravdava imputaciju umesto brisanja
nepotpunih redova.

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

### Unakrsna validacija

Prosek ± standardna devijacija preko 50 foldova, sortirano po ROC-AUC:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| SVM | 0.7704 ± 0.0522 | 0.6999 ± 0.1075 | 0.6467 ± 0.1134 | 0.6608 ± 0.0722 | 0.8476 ± 0.0432 | 0.7597 ± 0.0767 |
| XGBoost | 0.7656 ± 0.0569 | 0.6423 ± 0.0765 | 0.7687 ± 0.0898 | 0.6962 ± 0.0661 | 0.8447 ± 0.0479 | 0.7545 ± 0.0750 |
| Logistic Regression | 0.7581 ± 0.0548 | 0.6475 ± 0.0857 | 0.7048 ± 0.0829 | 0.6707 ± 0.0662 | 0.8441 ± 0.0435 | 0.7528 ± 0.0795 |
| Random Forest | 0.7648 ± 0.0443 | 0.7112 ± 0.0945 | 0.5680 ± 0.0969 | 0.6249 ± 0.0745 | 0.8407 ± 0.0482 | 0.7474 ± 0.0748 |
| KNN | 0.7702 ± 0.0425 | 0.7186 ± 0.0893 | 0.5774 ± 0.0895 | 0.6347 ± 0.0703 | 0.8382 ± 0.0467 | 0.7384 ± 0.0741 |
| Decision Tree | 0.7233 ± 0.0491 | 0.5966 ± 0.0815 | 0.7520 ± 0.1677 | 0.6474 ± 0.0781 | 0.8025 ± 0.0530 | 0.6557 ± 0.0709 |

### Zaključan test skup (n = 154)

| Model | Imputacija | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| XGBoost | MICE | 0.7338 | 0.5970 | 0.7407 | 0.6612 | 0.8207 | 0.6746 |
| SVM | MICE | 0.6883 | 0.5682 | 0.4630 | 0.5102 | 0.8172 | 0.6926 |
| Logistic Regression | MICE | 0.7208 | 0.5846 | 0.7037 | 0.6387 | 0.8104 | 0.6654 |
| Random Forest | KNN | 0.7208 | 0.6170 | 0.5370 | 0.5743 | 0.8091 | 0.6713 |
| KNN | MICE | 0.7468 | 0.6667 | 0.5556 | 0.6061 | 0.8046 | 0.6938 |
| Decision Tree | KNN | 0.7208 | 0.5753 | 0.7778 | 0.6614 | 0.7936 | 0.6240 |

Rezultati na test skupu su sistematski niži za oko 0.03 ROC-AUC
jedinica u odnosu na kros-validaciju. Razlog je optimistička
pristrasnost: ista procedura korišćena je i za izbor hiperparametara
i za izveštavanje, pa je nepristrasna ocena ona dobijena na
zaključanom test skupu.

### Odziv i propušteni slučajevi

Kod ovog problema odziv je posebno važan, jer lažno negativan nalaz
znači nedijagnostikovanog pacijenta. Od ukupno 54 pozitivna slučaja
u test skupu, modeli propuštaju:

| Model | Propušteno | Uhvaćeno | Lažnih uzbuna |
|---|---|---|---|
| Decision Tree | 12 | 42 | 31 |
| XGBoost | 14 | 40 | 27 |
| Logistic Regression | 16 | 38 | 27 |
| KNN | 24 | 30 | 15 |
| Random Forest | 25 | 29 | 18 |
| SVM | 29 | 25 | 19 |

Razlika nije posledica kvaliteta rangiranja nego položaja praga
odlučivanja. SVM je drugi po ROC-AUC, ali mu pretraga nije dodelila
balansiranje klasa (`class_weight=None`), pa pri podrazumevanom
pragu od 0.5 propušta više od polovine pozitivnih slučajeva.
XGBoost je jedini model sa istovremeno visokim odzivom i
prihvatljivom preciznošću, pa je po F1 meri najbolji.

### Statistička značajnost razlika

Friedmanov test pokazuje značajnu razliku među modelima
(χ² = 48.58, p < 0.001). Prosečni rangovi (1 = najbolji):

| Model | Prosečan rang |
|---|---|
| SVM | 2.81 |
| Logistic Regression | 3.03 |
| XGBoost | 3.11 |
| Random Forest | 3.42 |
| KNN | 3.54 |
| Decision Tree | 5.09 |

Naknadno poređenje Wilcoxonovim testom uz Holmovu korekciju pokazuje
da je razlika u potpunosti posledica slabijih performansi stabla
odlučivanja: značajno je 5 od 15 parova, a svih pet uključuje stablo.
Nijedna razlika među preostalih pet modela nije statistički značajna
(p > 0.08).

Holmova korekcija je pri tome bila neophodna — bez nje bi dva para
(KNN–SVM, Random Forest–SVM) bila pogrešno proglašena značajnim.

### Važnost atributa

SHAP analiza nad najboljim modelom daje sledeći poredak (srednja
apsolutna SHAP vrednost): Glucose 0.69, Insulin 0.49, BMI 0.43,
Age 0.30, DiabetesPedigreeFunction 0.23, Pregnancies 0.12,
SkinThickness 0.09, BloodPressure 0.04.

Poredak se poklapa sa nalazima eksplorativne analize, gde je glukoza
imala najveću korelaciju sa ishodom (0.49), a krvni pritisak
najmanje razdvajao klase.
