import matplotlib
import math
import matplotlib.pyplot as plt

import ssl

matplotlib.use('TkAgg')

ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784', version=1, cache=True, return_X_y=True)
y = y.astype(int)

# print('Data size: {0} x {1} and label size {2}'.format(X.shape[0], X.shape[1], y.shape[0]))
# print('The images are of size: {0} x {0}'.format(math.sqrt(X.shape[1])))
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
#
# print(" The Number of Training data is ", X_train.shape[0])
# print(" The Number of Testing data is ", X_test.shape[0])
#
#
# ##############==========[training]=============#######
#
MLP_model = MLPClassifier(hidden_layer_sizes=(200, 400, 150, 100), max_iter=2,  activation='relu',
                          verbose=True,)
#
MLP_model.fit(X_train, y_train)

print(MLP_model.n_layers_)
#
print(MLP_model.hidden_layer_sizes)
for layer in range(MLP_model.n_layers_ - 1):
    print("Layer ", layer, "Weights ", MLP_model.coefs_[layer].shape, "Bias is ", MLP_model.intercepts_[layer].shape)
# ##############==========[prediction]=============#######

text_7 = X_test[y_test == 7].to_numpy()
inst = text_7[0]
print(inst.shape)
dim = int(math.sqrt(inst.shape[0]))
image = inst.reshape(dim, dim)
# print(image.shape)
plt.imshow(image)
plt.show()
y_pred_test = MLP_model.predict([inst])
#
print(y_pred_test)
#
# ##############==========[evulation]=============#######
#
ac = accuracy_score(y_test, MLP_model.predict(X_test))

print(ac)
#
# # %%
#
# print(classification_report(y_test, MLP_model.predict(X_test)))
