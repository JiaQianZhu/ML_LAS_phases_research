import pandas as pd
import pickle

df = pd.read_csv('data/sciglass_data_wt.csv', index_col=False)

X = df.iloc[:, 0:29]
y = df.iloc[:, 29:]

features = ['Li2O', 'SiO2', 'Al2O3', 'ZnO', 'MgO', 'Na2O', 'K2O', 'TiO2', 'ZrO2', 'P2O5', 'As2O3', 'B2O3', 'BaO', 'CaO',
            'SrO', 'SnO2', 'Fe2O3', 'MnO2', 'CoO', 'CeO2', 'V2O5', 'Sb2O3', 'Nd2O3', 'Cr2O3', 'MoO3', 'F', 'La2O3',
            'Ta2O5', 'Cl']
targets = ['LD', 'β-spodumene', 'β-quartz s.s', 'LM', 'Keatite', 'Petalite', 'LiAlSi3O8', 'LiAlSi2O6', 'KMK',
           'Cristobalite', 'Li4SiO4', 'Mullite', 'Tridymite']

with open('multi_output_rf_model.pickle', 'rb') as file:
    rf_model = pickle.load(file)


def predict_probabilities_for_features(features):

    with open('multi_output_rf_model.pickle', 'rb') as file:
        rf_model = pickle.load(file)

    features_vector = [features]

    probabilities = rf_model.predict_proba(features_vector)

    return probabilities


results = []

for index, row in X.iterrows():
    features_values = row.tolist()
    predicted_probabilities = predict_probabilities_for_features(features_values)

    row_labels = []
    # Process each set of predicted probabilities
    for i, prob_array in enumerate(predicted_probabilities):
        # Check if prob_array has two elements and the second element is greater than or equal to 0.3
        if len(prob_array[0]) == 2 and prob_array[0, 1] >= 0.3:
            predicted_label = targets[i]

            row_labels.append(predicted_label)

    results.append(row_labels)


modified_data = [[','.join(inner_list)] for inner_list in results]
columns = ['predicted crystals']

df = pd.DataFrame(modified_data, columns=columns)

result = pd.concat([X, df], axis=1)


result.to_csv('data/predicted_crystals_sciglass.csv', index=False)
