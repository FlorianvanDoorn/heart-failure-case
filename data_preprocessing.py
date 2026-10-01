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

# Verwijder de kolommen die niet nodig zijn voor de analyse.
aantal_kolommen_voor = len(dataset.columns)
dataset = dataset.drop(columns=['CaseNumber', 'LastName', 'PostCode']) # verwijder de case number, last name en postcode kolommen
print("Verwijderde kolommen die niet nodig zijn voor de analyse:", aantal_kolommen_voor - len(dataset.columns))

# Converteer de HeartDisease-kolom naar 0 en 1
labels = dataset['HeartDisease'].astype(str).str.strip().str.lower() # verwijder spaties en zet de waarden om naar kleine letters
aantal_gecorrigeerde_labels = int((labels == 'yes').sum())
dataset['HeartDisease'] = pd.to_numeric(
    labels.replace({'yes': '1'}),
    errors='raise',
).astype(int)
if not dataset['HeartDisease'].isin([0, 1]).all():
    raise ValueError('HeartDisease moet alleen de waarden 0 of 1 bevatten.')
print("Labels omgezet van yes naar 1:", aantal_gecorrigeerde_labels) # print het aantal labels dat is omgezet van yes naar 1

# Verwijder rijen met ontbrekende waarden
aantal_voor = len(dataset) # Tel het aantal rijen in de dataset voordat we rijen verwijderen
dataset = dataset.dropna(subset=["Age"]) # Verwijder rijen waar de Age-kolom ontbreekt 
print("Verwijderde rijen met ontbrekende leeftijd:", aantal_voor - len(dataset)) # Print het aantal verwijderde rijen

# Verwijder rijen waarin de bloeddruk 0 of lager is, omdat dit niet realistisch is
aantal_voor = len(dataset)
dataset = dataset[dataset["RestingBP"] > 0]
print("Verwijderde rijen met bloeddruk gelijk aan of lager dan 0:", aantal_voor - len(dataset))

aantal_voor = len(dataset)
dataset = dataset[dataset["Sex"].isin(["M", "F"])] # Behoud rijen waarin Sex "M" of "F" is
print("Verwijderde rijen met ongeldige geslachtswaarde:", aantal_voor - len(dataset))

# Verdeel per blok van tien rijen voordat imputers of encoders worden gefit.
train_mask = np.arange(len(dataset)) % 10 < 8
train_records = dataset.iloc[train_mask].copy()
test_records = dataset.iloc[~train_mask].copy()

X_train = train_records.drop(columns=['HeartDisease'])
y_train = train_records['HeartDisease'].to_numpy()
X_test = test_records.drop(columns=['HeartDisease'])
y_test = test_records['HeartDisease'].to_numpy()

# Bereken de cholesterolmediaan apart voor training en test.
from sklearn.impute import SimpleImputer
train_imputer = SimpleImputer(missing_values=0, strategy='median')
X_train[['Cholesterol']] = train_imputer.fit_transform(X_train[['Cholesterol']])
X_test[['Cholesterol']] = train_imputer.transform(X_test[['Cholesterol']]) # Gebruik dezelfde mediaan als voor training, maar pas toe op test.
print('Mediaan cholesterol training:', train_imputer.statistics_[0])
print('Mediaan cholesterol test:', train_imputer.statistics_[0])

# Fit de encoder alleen op training en gebruik dezelfde kolommen voor test.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorische_kolommen = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorische_kolommen)],
    remainder='passthrough',
    sparse_threshold=0,
)
X_train_encoded = np.asarray(ct.fit_transform(X_train))
X_test_encoded = np.asarray(ct.transform(X_test))
kolomnamen = ct.get_feature_names_out(X_train.columns.tolist())

train_data = pd.DataFrame(X_train_encoded, columns=kolomnamen, index=X_train.index)
train_data['HeartDisease'] = y_train
test_data = pd.DataFrame(X_test_encoded, columns=kolomnamen, index=X_test.index)
test_data_met_uitkomst = test_data.copy()
test_data_met_uitkomst['HeartDisease'] = y_test

encoder_kolommen = [naam for naam in kolomnamen if naam.startswith('encoder__')]
train_data[encoder_kolommen] = train_data[encoder_kolommen].astype(int)
test_data[encoder_kolommen] = test_data[encoder_kolommen].astype(int)
test_data_met_uitkomst[encoder_kolommen] = test_data_met_uitkomst[encoder_kolommen].astype(int)
integer_kolommen = [
    'remainder__Age',
    'remainder__RestingBP',
    'remainder__Cholesterol',
    'remainder__FastingBS',
    'remainder__MaxHR',
]
train_data[integer_kolommen] = train_data[integer_kolommen].round().astype(int)
test_data[integer_kolommen] = test_data[integer_kolommen].round().astype(int)
test_data_met_uitkomst[integer_kolommen] = test_data_met_uitkomst[integer_kolommen].round().astype(int)

# Training bevat X en y; test bevat alleen X.
train_data.to_csv(bestand.with_name('Heart_failure_train.csv'), index=False)
test_data.to_csv(bestand.with_name('Heart_failure_test.csv'), index=False)
test_data_met_uitkomst.to_csv(bestand.with_name('Heart_failure_test_control.csv'), index=False)


