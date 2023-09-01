import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Déterminer le répertoire du script
from util import gen_tf2_pgm

script_dir = os.path.dirname(os.path.realpath(__file__))

# Charger le jeu de données CICIDS2017 au format CSV
data_path = os.path.join(script_dir, 'archive/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data = pd.read_csv(data_path, skipinitialspace=True)

# Enlever la première ligne (header) de X_data
data = data.iloc[1:]

# Remplacer les valeurs manquantes (NaN) par la moyenne des colonnes respectives
data = data.fillna(data.mean())

# Supprimer les lignes contenant des valeurs infinies
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Extraire les fonctionnalités (X) et les étiquettes (y)
X_data = data.drop(columns=["Label"])
y_labels = data['Label']

# Convertir les étiquettes en valeurs numériques
y_labels = y_labels.apply(lambda label: 0 if label == "BENIGN" else 1)

# Utiliser RobustScaler pour normaliser les données
scaler = RobustScaler()
X_data = scaler.fit_transform(X_data)

# Fractionner les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Créer le modèle
model = Sequential([
    Dense(200, input_dim=X_train.shape[1], activation='relu'),
    Dense(500, activation='relu'),
    Dense(200, activation='relu'),
    Dense(2, activation='softmax')  # Assurez-vous d'ajuster le nombre de classes ici
])

# Compiler le modèle
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=10,
                    batch_size=64, verbose=1,
                    validation_split=0.2)

# Fonction FGSM pour générer des exemples adversariaux
# def generate_adversarial_example(model, input_example, true_label, epsilon):
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#     with tf.GradientTape() as tape:
#         tape.watch(input_example)
#         prediction = model(input_example)
#         loss = loss_object(true_label, prediction)
#     gradient = tape.gradient(loss, input_example)
#     perturbed_example = input_example + epsilon * tf.sign(gradient)
#     return perturbed_example

def evaluation_metrics(y_test, y_pred):
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("recall_score: ", recall_score(y_test, y_pred, average='weighted'))
    print("precision_score: ", precision_score(y_test, y_pred, average='weighted'))
    print("f1_score: ", f1_score(y_test, y_pred, average='weighted'))
    print("{:.3f}".format(accuracy_score(y_test, y_pred)), " & ", "{:.3f}".format(recall_score(y_test, y_pred, average='weighted')), " & ", "{:.3f}".format(precision_score(y_test, y_pred, average='weighted')), " & ", "{:.3f}".format(f1_score(y_test, y_pred, average='weighted')))
    return accuracy_score(y_test, y_pred)

# attack_functions = [gen_tf2_fgsm_attack, gen_tf2_bim, gen_tf2_pgm] #, gen_tf2_mim]
attack_functions = [gen_tf2_pgm]
label_attack = ["fgsm", "bim", "pgm"]
# label_attack = ["pgm"]
# epsilons = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
epsilons = [3.5, 4, 4.5, 5]

accuracies = []

for attack_function in attack_functions:
    print("<" * 15, attack_function, ">" * 15)
    # epsilon = 0.1
    acc = []
    for epsilon in epsilons:
    # for epsilon in [1]:
        print("-" * 35)
        print("eps: ", epsilon)
        # Convertir y_test en tableau NumPy pour les vraies étiquettes
        y_test_np = y_test.to_numpy()

        # Convertir les données d'entrée en tenseurs TensorFlow
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test_tf = tf.convert_to_tensor(y_test_np, dtype=tf.int32)

        # Générer des exemples adversariaux pour le jeu de test
        perturbed_examples = attack_function(model, X_test_tf, epsilon)

        # Prédiction sur les exemples originaux et perturbés
        original_predictions = model.predict(X_test_tf)

        perturbed_predictions = np.argmax(model.predict(perturbed_examples), axis=1)
        # perturbed_predictions = model.predict(perturbed_examples)
        score = evaluation_metrics(perturbed_predictions, y_test)
        acc.append(score)
    accuracies.append(acc)

exit()

# Créer une figure et un axe
fig, ax = plt.subplots()

# Générer une courbe pour chaque attaque
color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

for i, attack in enumerate(label_attack):
    attack_accuracies = accuracies[i]
    ax.plot(epsilons, attack_accuracies, label=attack, color=color[i])


# Ajouter les labels et la légende
ax.set_xlabel('Epsilon value')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy as a function of Epsilon value for the {} attack method'.format(label_attack[0]))
ax.legend()

# Afficher le graphique
plt.show()

# # Afficher les résultats
# for i in range(len(X_test)):
#     original_pred_class = np.argmax(original_predictions[i])
#     perturbed_pred_class = np.argmax(perturbed_predictions[i])
#     print("Exemple", i+1)
#     print("Prédiction originale:", original_pred_class)
#     print("Confiance originale:", original_predictions[i][original_pred_class])
#     print("Prédiction perturbée:", perturbed_pred_class)
#     print("Confiance perturbée:", perturbed_predictions[i][perturbed_pred_class])
#     print("\n")



# ... (le reste du code pour afficher les graphiques, si nécessaire)
# Préparer les données pour le graphique
# labels = ['Classe ' + str(i) for i in range(len(original_predictions[0]))]
# original_probs = original_predictions[0]
# perturbed_probs = perturbed_predictions[0]
#
# # Créer un graphique à barres pour les probabilités originales et perturbées
# fig, ax = plt.subplots()
# x = np.arange(len(labels))
# width = 0.35
#
# rects1 = ax.bar(x - width/2, original_probs, width, label='Origine')
# rects2 = ax.bar(x + width/2, perturbed_probs, width, label='Perturbé')
#
# ax.set_ylabel('Probabilité')
# ax.set_title('Probabilités de prédictions originales et perturbées')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# # Ajouter les étiquettes des probabilités au-dessus des barres
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('%.2f' % height,
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points de décalage
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
#
# fig.tight_layout()
#
# # Afficher le graphique
# plt.show()