from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / 'Heart_failure_train.csv'
TEST_PATH = PROJECT_DIR / 'Heart_failure_test.csv'
TEST_CONTROL_PATH = PROJECT_DIR / 'Heart_failure_test_control.csv'
MODEL_PATH = PROJECT_DIR / 'gradient_boosting_model.joblib'
PREDICTIONS_PATH = PROJECT_DIR / 'GradientBoosting_test_with_predictions.csv'


def main() -> None:
    # Lees de train-, test- en controlebestanden in.
    training_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    test_control = pd.read_csv(TEST_CONTROL_PATH)

    # X bevat de kenmerken; y is het label dat het model leert voorspellen.
    X = training_data.drop(columns=['HeartDisease'])
    y = training_data['HeartDisease']
    X_test = test_data.copy()
    y_test = test_control['HeartDisease']

    # Controleer dat training en test dezelfde kenmerken en rijen gebruiken.
    if list(X.columns) != list(X_test.columns):
        raise ValueError('De featurekolommen van de train- en testset komen niet overeen.')
    if not X_test.reset_index(drop=True).equals(
        test_control.drop(columns=['HeartDisease']).reset_index(drop=True)
    ):
        raise ValueError('De testset en het testcontrolebestand bevatten niet dezelfde rijen.')

    # Bouw bomen na elkaar op; elke nieuwe boom probeert eerdere fouten te corrigeren.
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)

    # Vergelijk de voorspellingen met de echte uitkomsten uit het controlebestand.
    predicted_labels = model.predict(X_test)
    # Haal naast het voorspelde label ook de kans op klasse 1 (hartziekte) op.
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predictions = X_test.copy()
    predictions['HeartDisease'] = y_test.to_numpy()
    predictions['PredictedHeartDisease'] = predicted_labels
    predictions['ProbabilityHeartDisease'] = probabilities

    # Rapporteer hoe vaak de voorspelling klopt en waar fouten vallen.
    print(f'Test-accuracy: {accuracy_score(y_test, predicted_labels):.3f}')
    print('Classificatierapport:')
    print(classification_report(y_test, predicted_labels, labels=[0, 1], target_names=['Geen hartziekte', 'Hartziekte'], zero_division=0))
    print('Confusion matrix (werkelijk, voorspeld):')
    print(confusion_matrix(y_test, predicted_labels, labels=[0, 1]))

    # Bewaar resultaten en het getrainde model voor later gebruik.
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    print(f'Voorspellingen opgeslagen in: {PREDICTIONS_PATH.name}')
    print(f'Model opgeslagen in: {MODEL_PATH.name}')


if __name__ == '__main__':
    main()