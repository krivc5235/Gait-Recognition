import os
import numpy as np
import cv2
import numpy as np

import os
import scipy.misc
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.svm import LinearSVC
import time
from sklearn.utils import shuffle

data = r"C:\Users\jankr\Documents\Jan\Faks\magisterij\Drugi\IOI\GaitRecognition\gei_test"

X_train =[]
X_test=[]
y_train =[]
y_test=[]

def prepare_data(folder):
    x = []
    y = []
    for person in os.listdir(folder):
        if person != 'test':
            label = (person[-7:-4])
            image = cv2.imread(os.path.join(folder, person))[:, :, 0]
            x.append(image)
            y.append(label)
    return x, y

view_list = os.listdir(data)
view_list.sort()
# Walk the input path
for view in view_list:

    path = os.path.join(data, view)
    X_train, y_train = prepare_data(path)
    X_test, y_test = prepare_data(os.path.join(path, 'test'))

    X_test = np.array(X_test)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
    print("X_test.shape: {}, y_test.shape: {}".format(X_test.shape, y_test.shape))

    tic_whole = time.time()

    X_train=X_train.reshape(X_train.shape[0],128*88)
    X_test=X_test.reshape(X_test.shape[0],128*88)


    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)


    tic=time.time()
    pca=PCA(n_components=0.99)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    toc=time.time()
    print("Viewing angle: ", view)
    #print("Dimensionality reduced to:", pca.n_components_)
    #print("Time taken to reduce the size of the data : "+str((toc-tic))+" sec")

    clf = LogisticRegression(C=.01,solver ='lbfgs',multi_class='auto',max_iter=250)
    clf = clf.fit(X_train_pca, y_train)
    y_train_pred=clf.predict(X_train_pca)
    y_test_pred=clf.predict(X_test_pca)
    score = 0
    for i in range (len(X_test)):
        if y_test[i] == y_test_pred[i]:
            score += 1
    #print(score,  len(X_test))
    #print("Lastni score: ", score / len(X_test))
    print(y_test_pred)
    print("Training accuracy: {:.4f}, Test Accuracy: {:.4f}".format(accuracy_score(y_train_pred, y_train), accuracy_score(y_test_pred, y_test)))

    toc_whole = time.time()

    print("Time taken for dimensionality reduction, fitting and testing : "+str((toc_whole-tic_whole))+" sec")
    print("------------------------------------------------------------------------------------------------")