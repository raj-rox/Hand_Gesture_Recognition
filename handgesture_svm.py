import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump, load
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from keras.utils import to_categorical

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

images_done = 0
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('../leapGestRecog/00/'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup
'''
x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('../leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('../leapGestRecog/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('../leapGestRecog/0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                grey_img = rgb2grey(arr)
                hog_features = hog(grey_img, block_norm='L2-Hys', pixels_per_cell=(16, 16))
                x_data.append(hog_features)
                count = count + 1
                images_done = images_done + 1
                print('../leapGestRecog/0' + str(i) + '/' + str(j) + '/' + str(k) + ' DONE!', str(images_done)+'/20000')
            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')

ss = StandardScaler()
x_data = ss.fit_transform(x_data)
pca = PCA(n_components=500)
x_data = pca.fit_transform(x_data)

y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

np.save('samples', x_data)
np.save('labels', y_data)'''

x_data = np.load('samples.npy')
y_data = np.load('labels.npy')

# y_data = to_categorical(y_data)

# y_data = y_data.reshape(20000, 10)

# x_data = x_data.reshape((20000, 120, 320, 1))
# x_data /= 255

x_train,x_test,y_train,y_test = train_test_split(x_data, y_data, test_size=0.2)
#x_validate,x_test,y_validate,y_test = train_test_split(x_further, y_further, test_size=0.5)

# clf = load('Model.joblib')

clf = SVC(verbose=True)
clf.fit(x_train, y_train)

dump(clf, 'Model.joblib')

y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
