import math

import matplotlib
import ssl

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from skimage import io,color
from skimage.transform import resize
import glob

matplotlib.use('TkAgg')
ssl._create_default_https_context = ssl._create_unverified_context
# ==================================[dataset]===========================#
path_g = "/Users/ahmedmadkour/Documents/faculty/2023-first term/python/dl/datasets/Sampleg/*"
path_o = "/Users/ahmedmadkour/Documents/faculty/2023-first term/python/dl/datasets/Sampleo/*"
# path_8 = "/Users/ahmedmadkour/Documents/faculty/2023-first term/python/dl/datasets/Sample8/*"
# path_q = "/Users/ahmedmadkour/Documents/faculty/2023-first term/python/dl/datasets/Sampleq/*"
# path_8_paths = glob.glob(path_8)
# path_q_paths = glob.glob(path_q)
path_g_paths = glob.glob(path_g)
print(len(path_g_paths))
path_o_paths = glob.glob(path_o)
print(len(path_o_paths))
X = []
y = []
# for path in path_8_paths:
#     img = io.imread(path)
#     img_resized = resize(img, (30, 30, 3))  # 30*30*3
#     img_gray = color.rgb2gray(img_resized)  # # 30*30
#     img_reshaped = img_gray.reshape(900)
#     X.append(img_reshaped)
#     y.append("8")
for path in path_o_paths:
    img = io.imread(path)
    img_resized = resize(img, (30, 30, 3))  # 30*30*3
    img_gray = color.rgb2gray(img_resized)  # # 30*30
    img_reshaped = img_gray.reshape(900)
    X.append(img_reshaped)
    y.append("o")
# for path in path_q_paths:
#     img = io.imread(path)
#     img_resized = resize(img, (30, 30, 3))  # 30*30*3
#     img_gray = color.rgb2gray(img_resized)  # # 30*30
#     img_reshaped = img_gray.reshape(900)
#     X.append(img_reshaped)
#     y.append("q")

for path in path_g_paths:
    img = io.imread(path)
    img_resized = resize(img, (30, 30, 3))  # 30*30*3
    img_gray = color.rgb2gray(img_resized) # # 30*30
    d=np.shape(img_gray)[0]
    img_reshaped = img_gray.reshape(d*d)
    X.append(img_reshaped)
    y.append("g")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=0)
#
# print(" The Number of Training data is ", X_train.shape[0])
# print(" The Number of Testing data is ", X_test.shape[0])
#
#
# ##############==========[training]=============#######
#
MLP_model = MLPClassifier(hidden_layer_sizes=(200, 400, 150, 100), max_iter=10, activation='relu',
                          verbose=True, )
#
MLP_model.fit(X, y)
print(MLP_model.n_layers_)
#
print(MLP_model.hidden_layer_sizes)
for layer in range(MLP_model.n_layers_ - 1):
    print("Layer ", layer, "Weights ", MLP_model.coefs_[layer].shape, "Bias is ", MLP_model.intercepts_[layer].shape)
# ##############==========[prediction]=============#######

# text_7 = X_test[y_test == 7].to_numpy()
text_o = X[y == "o"]
inst = text_o[1]
# new_dim=int(math.sqrt(inst.shape[0]))
# image=inst.reshape((new_dim,new_dim))
# plt.imshow(image)
# plt.show()

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
