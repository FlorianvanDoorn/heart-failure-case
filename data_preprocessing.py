# Data preprocessing for heart failure dataset

# Importeer de benodigde libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path




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

# Verwijder rijen met ontbrekende waarden
aantal_voor = len(dataset) # Tel het aantal rijen in de dataset voordat we rijen verwijderen
dataset = dataset.dropna(subset=["Age"]) # Verwijder rijen waar de Age-kolom ontbreekt 
print("Verwijderde rijen:", aantal_voor - len(dataset)) # Print het aantal verwijderde rijen

# Verwijder rijen waarin de bloeddruk 0 of lager is, omdat dit niet realistisch is
aantal_voor = len(dataset)
dataset = dataset[dataset["RestingBP"] > 0]
print("Verwijderde rijen met bloeddruk gelijk aan of lager dan 0:", aantal_voor - len(dataset))

aantal_voor = len(dataset)
dataset = dataset[dataset["Sex"].isin(["M", "F"])] # Behoud rijen waarin Sex "M" of "F" is
print("Verwijderde rijen met ongeldige geslachtswaarde:", aantal_voor - len(dataset))

# Splits de dataset in input (X) en output (y)
X = dataset.iloc[:, :-1].values # Alle kolommen behalve de laatste (HeartDisease) Dit is de input (features)
y = dataset.iloc[:, -1].values # Alleen de laatste kolom (HeartDisease) Dit is de output (target)

# Toon de input en output
print(X) # Toon de input (features)
print(y) # Toon de output (target)

# Imputeer nulwaarden in kolom 7 met de gemiddelde waarde van die kolom
from sklearn.impute import SimpleImputer # Importeer SimpleImputer uit sklearn
imputer = SimpleImputer(missing_values=0, strategy='mean') # Vervang nulwaarden door het kolomgemiddelde
imputer.fit(X[:, 7:8]) # Pas de SimpleImputer alleen toe op kolom 7
X[:, 7:8] = imputer.transform(X[:, 7:8]) # Vervang de nulwaarden door de gemiddelde waarde

# print(X[:, 7:8]) # Toon de input (features) na het invullen van ontbrekende waarden

# Encodeer categorische variabelen met OneHotEncoder
from sklearn.compose import ColumnTransformer # Importeer ColumnTransformer uit sklearn
from sklearn.preprocessing import OneHotEncoder # Importeer OneHotEncoder uit sklearn
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4, 5, 11])], remainder='passthrough') # Pas OneHotEncoder toe op kolommen 4, 5 en 11 (categorische variabelen) en laat de rest van de kolommen ongemoeid
X = np.array(ct.fit_transform(X)) # Transformeer de input (X) met de ColumnTransformer en converteer het naar een numpy-array

print(X) # Toon de input (features) na het encoderen van categorische variabelen

# Sla de bewerkte input en de uitkomst samen op voor controle.
# OneHotEncoder verandert het aantal en de volgorde van de kolommen.
# Vraag daarom de nieuwe kolomnamen op bij de ColumnTransformer.
kolomnamen = ct.get_feature_names_out(dataset.columns[:-1].tolist())
bewerkte_data = pd.DataFrame(X, columns=kolomnamen, index=dataset.index)
bewerkte_data['HeartDisease'] = y

# Sla op naast het bronbestand. De bewerkte CSV wordt overschreven.
bewerkte_data.to_csv(bestand.with_name('Heart_failure_bewerkt.csv'), index=False)

#test
#test beide uploaden Florian
