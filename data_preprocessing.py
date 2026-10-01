# Data preprocessing for heart failure dataset

# Importeer de benodigde libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path




mogelijke_paden = [
    Path(__file__).resolve().parent / 'Heart faillure prediction data.csv',
    Path('../data/Heart faillure prediction data.csv'),
    Path('heart-failure/data/Heart faillure prediction data.csv'),
    Path('data/Heart faillure prediction data.csv'),
]
bestand = next((pad for pad in mogelijke_paden if pad.is_file()), None)
if bestand is None:
    raise FileNotFoundError('CSV niet gevonden. Open dit notebook vanuit de map heart-failure/code.')

# Lees de CSV-data in
dataset = pd.read_csv(bestand)
dataset = dataset.drop(columns=['CaseNumber', 'LastName', 'PostCode'])

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
X = dataset.iloc[:, :-1].copy() # Alle kolommen behalve de laatste (HeartDisease) Dit is de input (features)
y = dataset.iloc[:, -1].values # Alleen de laatste kolom (HeartDisease) Dit is de output (target)

# Toon de input en output
print(X) # Toon de input (features)
print(y) # Toon de output (target)

# Imputeer nulwaarden in kolom 7 met de gemiddelde waarde van die kolom
from sklearn.impute import SimpleImputer # Importeer SimpleImputer uit sklearn
imputer = SimpleImputer(missing_values=0, strategy='median') # Vervang nulwaarden door het kolommediaan
X[['Cholesterol']] = imputer.fit_transform(X[['Cholesterol']]) # Vervang nullen in Cholesterol door de mediaan



# Encodeer categorische variabelen met OneHotEncoder
from sklearn.compose import ColumnTransformer # Importeer ColumnTransformer uit sklearn
from sklearn.preprocessing import OneHotEncoder # Importeer OneHotEncoder uit sklearn
categorische_kolommen = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorische_kolommen)], remainder='passthrough')
X = np.array(ct.fit_transform(X)) # Transformeer de input (X) met de ColumnTransformer en converteer het naar een numpy-array

print(X) # Toon de input (features) na het encoderen van categorische variabelen

# Sla de bewerkte input en de uitkomst samen op voor controle.
# OneHotEncoder verandert het aantal en de volgorde van de kolommen.
# Vraag daarom de nieuwe kolomnamen op bij de ColumnTransformer.
kolomnamen = ct.get_feature_names_out(dataset.columns[:-1].tolist())
bewerkte_data = pd.DataFrame(X, columns=kolomnamen, index=dataset.index)
bewerkte_data['HeartDisease'] = y
encoder_kolommen = [naam for naam in kolomnamen if naam.startswith('encoder__')]
bewerkte_data[encoder_kolommen] = bewerkte_data[encoder_kolommen].astype(int)
integer_kolommen = [
    'remainder__Age',
    'remainder__RestingBP',
    'remainder__Cholesterol',
    'remainder__FastingBS',
    'remainder__MaxHR',
]
bewerkte_data[integer_kolommen] = bewerkte_data[integer_kolommen].astype(int)

# Verdeel per blok van tien rijen: acht voor training en twee voor test.
train_mask = np.arange(len(bewerkte_data)) % 10 < 8
train_data = bewerkte_data.iloc[train_mask]
test_data = bewerkte_data.iloc[~train_mask].drop(columns=['HeartDisease'])

# Training bevat X en y; test bevat alleen X.
train_data.to_csv(bestand.with_name('Heart_failure_train.csv'), index=False)
test_data.to_csv(bestand.with_name('Heart_failure_test.csv'), index=False)



