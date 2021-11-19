import numpy as np
import KNN_real as kt
import KDbaum as kb
import os
import matplotlib.pyplot as plt
from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition as fr
import time

print("Training KNN classifier ~~~~")
X = []
Y = []
# Iterate through each person in the training set. "class_dir" is the name of the person obtained
for class_dir in os.listdir("examples/train"):
    if not os.path.isdir(os.path.join("examples/train", class_dir)):
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/train", class_dir)):
        try:
            image = fr.load_image_file(img_path)
            boxes = fr.face_locations(image)
            # Returns a vector of 128 dimensions
            X.append(fr.face_encodings(image, known_face_locations=boxes)[0])
            Y.append(class_dir)
        except:
            print('failed!')
print("Training finished !!!!!")
X_test = []
Y_test = []

for class_dir in os.listdir("examples/test"):
    if not os.path.isdir(os.path.join("examples/test", class_dir)):
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/test", class_dir)):
        try:
            image = fr.load_image_file(img_path)
            boxes = fr.face_locations(image)
            # Returns a vector of 128 dimensions
            X_test.append(fr.face_encodings(image, known_face_locations=boxes)[0])
            Y_test.append(class_dir)
        except:
            print('failed!')

#Facial feature set used for training
dataArray_train = np.array(X)
labelArray_train = np.array(Y)
#Facial feature set used for testing
dataArray_test = np.array(X_test)
labelArray_test = np.array(Y_test)


t1 = time.perf_counter()#Record the time to start building the kd tree

print("Building kd tree ~~~~~~~~")
kd_tree_all = kb.Kd_Tree(dataArray_train, labelArray_train)
print("Build finish!!!!!!!")

t2 = time.perf_counter()#Record the time when the kd tree is built

accuracies = []#list of accuracies
y_hat_test = []#Prediction results of knn algorithm without kd tree construction
y_hat_test_kd =[]#prediction result of the knn algorithm constructed with the kd tree

t_knn = 0
t_kd = 0
#Testing the effect of different k on the accuracy of knn

for k in range(1,len(Y)+1):
    i = 0
    count = 0  # right result of knn without kd tree
    count_kd = 0 # right result of knn with kd tree
    t3 = time.perf_counter()  # Brute force search start time

    for test_point in X_test:
        # Predict face identity
        predic = kt.knn_predict(X_train=X, X_test=test_point.reshape(1, -1), y_train=Y, k=k, p=1, threshold=5)[0]
        if Y_test[i] == predic:# if prediction is right
            count += 1
        i += 1
    t4 = time.perf_counter()  # Brute force search end time
    i = 0
    t5 = time.perf_counter() #kd tree search start time
    for test_point in X_test:#Traverse all faces to be predicted
        # Predict face identity
        label = kd_tree_all.knn_face_recognition(values=test_point, k=k, sigma=1)
        if Y_test[i] == label:# if prediction is right
            count_kd += 1
        # Forecast result storage
        i += 1
    t6 = time.perf_counter()  # kd tree search end time

    t_knn = t4 - t3
    t_kd = t6 - t5

    # Calculation Accuracy and time consuming
    print('knn algorithm without kd tree time consuming:{:.4f}s, bei k is '.format(t_knn), format(k))
    print('knn algorithm with kd tree time consuming:{:.4f}s, bei k is '.format(t_kd), format(k))
    print("Accuracy without kd tree: {:.4f} %, bei k :".format((count / len(Y_test)) * 100), format(k))
    print("Accuracy with kd tree: {:.2f} %, bei k : ".format((count_kd / len(Y_test)) * 100), format(k))
    accuracies.append(count_kd / len(Y_test))


# Plot the results

plt.plot(range(1,len(Y)+1), accuracies,label = "knn")
plt.xlabel('# of Nearest Neighbors (k)')
#plt.xlabel('# of minkowski_distance (p)')
#plt.xlabel('# of Weighted knn (sigma)')
plt.ylabel('Accuracy (%)')

plt.show()
print('Time consuming to build kd tree:{:.4f}s'.format(t2 - t1))


