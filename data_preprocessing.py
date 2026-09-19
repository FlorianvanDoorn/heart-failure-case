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

print(X[:, 7:8]) # Toon de input (features) na het invullen van ontbrekende waarden