# Data preprocessing for heart failure dataset

# Importeer de benodigde libraries
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split




mogelijke_paden = [
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
print("Verwijderde rijen:", aantal_voor - len(dataset)) # Print het aantal verwijderde rijen

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

# Bereken de cholesterolmediaan uitsluitend met niet-nulle waarden uit de trainingsset.
# Pas dezelfde mediaan toe op de testset, zonder opnieuw te fitten.
imputer = SimpleImputer(missing_values=0, strategy="median")
X_train["Cholesterol"] = imputer.fit_transform(X_train[["Cholesterol"]]).ravel()
X_test["Cholesterol"] = imputer.transform(X_test[["Cholesterol"]]).ravel()

# Selecteer categorieën op naam, zodat gewijzigde kolomposities geen probleem zijn.
categorische_kolommen = [
    "Sex", "ChestPainType", "FastingBS",
    "RestingECG", "ExerciseAngina", "ST_Slope"
]

# Encodeer categorieën en geef meetwaarden ongewijzigd door, zonder naamvoorvoegsels.
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorische_kolommen)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
# Leer de categorieën op de trainingsset; gebruik dezelfde encoder voor de testset.
# Een nieuwe categorie in de testset levert nullen in de bijbehorende encoderkolommen op.
train_waarden = ct.fit_transform(X_train)
test_waarden = ct.transform(X_test)

# Behoud de index zodat kenmerken en HeartDisease bij het samenvoegen correct uitlijnen.
kolomnamen = ct.get_feature_names_out()
X_train = pd.DataFrame(train_waarden, columns=kolomnamen, index=X_train.index)
X_test = pd.DataFrame(test_waarden, columns=kolomnamen, index=X_test.index)

# Exporteer de modelkenmerken met de bijbehorende uitkomst HeartDisease.
# Gebruik bij modeltraining X_train en y_train, niet de volledige exporttabel.
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
train_data.to_csv(bestand.with_name('Heart_failure_train.csv'), index=False)
test_data.to_csv(bestand.with_name('Heart_failure_test.csv'), index=False)

# Vernieuw ook het volledige controlebestand, in de oorspronkelijke rijvolgorde.
# Dit gecombineerde bestand is geen nieuwe trainingsset: houd de aparte testset apart.
bewerkte_data = pd.concat([train_data, test_data]).sort_index()
bewerkte_data.to_csv(bestand.with_name('Heart_failure_bewerkt.csv'), index=False)

# Toon de setgroottes en klasseverdeling om de splitsing te controleren.
print(f'Trainingsset: {len(X_train)} rijen; testset: {len(X_test)} rijen.')
print(pd.DataFrame({
    'Training': y_train.value_counts().sort_index(),
    'Test': y_test.value_counts().sort_index(),
}))

# Dit is de subbranch van Florian
