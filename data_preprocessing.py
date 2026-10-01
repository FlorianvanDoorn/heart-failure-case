# Data preprocessing for heart failure dataset

# Importeer de benodigde libraries
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix




# Laat None staan tijdens modelontwikkeling. Kies pas na cross-validatie
# "Logistische regressie" of "Lineaire SVM" voor de eindbeoordeling.
gekozen_naam = None

# Zoek eerst relatief aan dit script, onafhankelijk van de huidige werkmap.
mogelijke_paden = [
    Path(__file__).resolve().parent.parent / 'data' / 'Heart faillure prediction data.csv',
    Path('../data/Heart faillure prediction data.csv'),
    Path('heart-failure/data/Heart faillure prediction data.csv'),
    Path('data/Heart faillure prediction data.csv'),
]
bestand = next((pad for pad in mogelijke_paden if pad.is_file()), None)
if bestand is None:
    raise FileNotFoundError('CSV niet gevonden. Open dit notebook vanuit de map heart-failure/code.')

# Lees de CSV-data in
dataset = pd.read_csv(bestand)

# vervang de waarden in de kolom HeartDisease van "yes" naar 1
print("Aantal rijen met HeartDisease = 'yes':", dataset[dataset["HeartDisease"] == "yes"].shape[0]) # print het aantal rijen waar als heartdisease "yes" is, zodat we kunnen zien welke rijen we gaan vervangen
dataset["HeartDisease"] = dataset["HeartDisease"].replace({"yes": "1"}).astype(int) # Zet alle uitkomstcodes om naar gehele getallen.
print("Aantal rijen met HeartDisease = 'yes' na vervanging:", dataset[dataset["HeartDisease"] == "yes"].shape[0]) # print het aantal rijen waar als heartdisease "yes" is, zodat we kunnen zien dat de vervanging is gelukt

# Verwijder rijen met ontbrekende waarden
aantal_voor = len(dataset) # Tel het aantal rijen in de dataset voordat we rijen verwijderen
dataset = dataset.dropna(subset=["Age"]) # Verwijder rijen waar de Age-kolom ontbreekt 
print("Verwijderde rijen met ontbrekende waarden in Age:", aantal_voor - len(dataset)) # Print het aantal verwijderde rijen

# Verwijder rijen waarin de bloeddruk 0 is
aantal_voor = len(dataset)
dataset = dataset[dataset["RestingBP"] > 0]
print("Verwijderde rijen met bloeddruk gelijk aan of lager dan 0:", aantal_voor - len(dataset))

# verwijder rijen met een ongeldige waarde in de kolom Sex (alleen "M" of "F" zijn geldig)
aantal_voor = len(dataset)
dataset = dataset[dataset["Sex"].isin(["M", "F"])] # Behoud rijen waarin Sex "M" of "F" is
print("Verwijderde rijen met ongeldige geslachtswaarde:", aantal_voor - len(dataset))

# Selecteer modelkenmerken als tabel met kolomnamen, zonder persoonsgegevens.
X = dataset.drop(columns=["CaseNumber", "LastName", "PostCode", "HeartDisease"])
y = dataset["HeartDisease"]

##### - geen rijen meer verwijderen na deze stap - ####

# Splits vóór het leren van de mediaan en de categorieën: 80% training en 20% test.
# stratify bewaart ongeveer de klasseverhouding; random_state maakt de verdeling herhaalbaar.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Selecteer categorieën op naam voor one-hot encoding binnen elke fold.
categorische_kolommen = [
    "Sex", "ChestPainType", "FastingBS",
    "RestingECG", "ExerciseAngina", "ST_Slope"
]

# Behandel cholesterol apart: alleen daar geldt nul als ontbrekende meting.
overige_numerieke_kolommen = ["Age", "RestingBP", "MaxHR", "Oldpeak"]
cholesterol_pipeline = Pipeline([
    ("imputatie", SimpleImputer(missing_values=0, strategy="median")),
    ("schaling", StandardScaler()),
])

# Definieer de voorbewerking zonder al iets op de volledige trainingsset te leren.
voorbewerking = ColumnTransformer(
    transformers=[
        ("cholesterol", cholesterol_pipeline, ["Cholesterol"]),
        ("numeriek", StandardScaler(), overige_numerieke_kolommen),
        ("categorie", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         categorische_kolommen),
    ],
    remainder="drop",
)

# Geef elk model een onafhankelijke, nog niet getrainde voorbewerking.
pipelines = {
    "Logistische regressie": Pipeline([
        ("voorbewerking", clone(voorbewerking)),
        ("model", LogisticRegression(max_iter=2000)),
    ]),
    "Lineaire SVM": Pipeline([
        ("voorbewerking", clone(voorbewerking)),
        ("model", LinearSVC(max_iter=10000, random_state=42)),
    ]),
}

# Controleer een eventuele modelkeuze voordat de berekeningen starten.
if gekozen_naam is not None and gekozen_naam not in pipelines:
    raise ValueError(f'Kies None of een van deze modelnamen: {list(pipelines)}')

# Leg dezelfde vijf gestratificeerde verdelingen vast voor beide modellen.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train))

# Precision, recall en F1 betreffen de positieve klasse HeartDisease = 1.
# ROC-AUC gebruikt modelscores en vereist daarom geen SVM-kanskalibratie.
metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
scores_per_model = {}
samenvatting = []

# Train per fold de volledige pipeline opnieuw, uitsluitend op de trainingsfolds.
for naam, pipeline in pipelines.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=folds, scoring=metrics,
        error_score="raise",
    )
    scores_per_model[naam] = scores

    # De sleutels test_* betreffen hier validatiefolds, niet de aparte testset.
    rij = {"Model": naam}
    for metric in metrics:
        fold_scores = scores[f"test_{metric}"]
        rij[f"{metric}_gemiddeld"] = fold_scores.mean()
        rij[f"{metric}_std"] = fold_scores.std()
    samenvatting.append(rij)

# Bewaar numerieke resultaten voor verdere analyse; formatteer alleen de uitvoer.
resultaten = pd.DataFrame(samenvatting).set_index("Model")

# Zet metrics onder elkaar, zodat de vergelijking in een gewone terminal past.
# Iedere cel bevat het gemiddelde en de standaardafwijking met drie decimalen.
metric_labels = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}
label_breedte = max(len(label) for label in metric_labels.values())
model_breedte = max(17, max(len(naam) for naam in resultaten.index))
kop = f'{"Metric":<{label_breedte}} | ' + ' | '.join(
    f'{naam:>{model_breedte}}' for naam in resultaten.index
)
print('\nCross-validatie (5 folds)')
print('Scores: gemiddelde +/- standaardafwijking (schaal 0-1)\n')
print(kop)
print('-' * len(kop))
for metric, label in metric_labels.items():
    # Gebruik vaste kolombreedtes en rechts uitgelijnde getallen.
    cellen = []
    for naam in resultaten.index:
        gemiddelde = resultaten.loc[naam, f'{metric}_gemiddeld']
        spreiding = resultaten.loc[naam, f'{metric}_std']
        cel = f'{gemiddelde:.3f} +/- {spreiding:.3f}'
        cellen.append(f'{cel:>{model_breedte}}')
    print(f'{label:<{label_breedte}} | ' + ' | '.join(cellen))

# Exporteer geen vooraf getransformeerde train/test-CSV's: de pipeline heeft
# oorspronkelijke kenmerken nodig. Bestaande CSV-exports worden niet vernieuwd.

# Beoordeel de testset alleen wanneer vooraf expliciet een model is gekozen.
if gekozen_naam is None:
    print('\nTestset niet beoordeeld. Stel gekozen_naam pas in na je modelkeuze.')
else:
    # Leer de gekozen pipeline opnieuw op de volledige trainingsset.
    definitief_model = clone(pipelines[gekozen_naam])
    definitief_model.fit(X_train, y_train)

    # Gebruik voor de testset uitsluitend de zojuist geleerde voorbewerking.
    y_voorspeld = definitief_model.predict(X_test)
    print(f'\nEindbeoordeling: {gekozen_naam}')
    print(classification_report(
        y_test, y_voorspeld, labels=[0, 1],
        target_names=["Geen hartaandoening", "Hartaandoening"],
        digits=3,
        zero_division=0,
    ))

    # Rijen tonen werkelijke klassen; kolommen tonen voorspelde klassen.
    matrix = confusion_matrix(y_test, y_voorspeld, labels=[0, 1])
    # Druk de volledige matrix af zonder afkorting of automatische regelomloop.
    print('Confusion matrix (aantallen; 0 = geen aandoening, 1 = aandoening)')
    print(pd.DataFrame(
        matrix,
        index=["Werkelijk 0", "Werkelijk 1"],
        columns=["Voorspeld 0", "Voorspeld 1"],
    ).to_string())



# Dit is de subbranch van Florian
