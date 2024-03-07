import pickle

features = ['Li2O', 'SiO2', 'Al2O3', 'ZnO', 'MgO', 'Na2O', 'K2O', 'TiO2', 'ZrO2', 'P2O5', 'As2O3', 'B2O3',
            'BaO', 'CaO',
            'SrO', 'SnO2', 'Fe2O3', 'MnO2', 'CoO', 'CeO2', 'V2O5', 'Sb2O3', 'Nd2O3', 'Cr2O3', 'MoO3', 'F',
            'La2O3',
            'Ta2O5', 'Cl']
targets = ['LD', 'β-spodumene', 'β-quartz s.s', 'LM', 'Keatite', 'Petalite', 'LiAlSi3O8', 'LiAlSi2O6', 'KMK',
           'Cristobalite', 'Li4SiO4', 'Mullite', 'Tridymite']


def predict_labels_for_features(features):
    # load the model
    with open('rf_model.pickle', 'rb') as file:
        rf_model = pickle.load(file)

    features_vector = [features]

    # calculate probabilities
    probabilities = rf_model.predict_proba(features_vector)

    # predict possible labels
    predicted_labels = [targets[i] for i, prob in enumerate(probabilities[0]) if any(prob > 0.5)]

    return predicted_labels


# input the desired compositions
def get_input():
    features_input = []
    for feature in features:
        ratio = float(input(f"Enter the weight ratio of {feature}: "))
        features_input.append(ratio)
    return features_input



def main():

    features_input = get_input()

    predicted_labels = predict_labels_for_features(features_input)

    print("Predicted crystal phases:")
    print(predicted_labels)


if __name__ == "__main__":
    main()
