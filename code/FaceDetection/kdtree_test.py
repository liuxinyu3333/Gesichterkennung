import numpy as np
import KNN_real as kt
import KDbaum as kb
import os

from face_recognition.face_recognition_cli import image_files_in_folder
import face_recognition as fr
import time

print("Training KNN classifier ~~~~")
X = []
Y = []
# Iterate through each person in the training set. "class_dir" is the name of the person obtained
for class_dir in os.listdir("examples/Harrypotter"):
    if not os.path.isdir(os.path.join("examples/Harrypotter", class_dir)):
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/Harrypotter", class_dir)):
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

for class_dir in os.listdir("examples/Harrypotter_test"):
    if not os.path.isdir(os.path.join("examples/Harrypotter_test", class_dir)):
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/Harrypotter_test", class_dir)):
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

accuracies = []
y_hat_test = []#Prediction results of knn algorithm without kd tree construction
y_hat_test_kd =[]#prediction result of the knn algorithm constructed with the kd tree

t3 = time.perf_counter()#Brute force search start time

for test_point in X_test:
    y_hat_test.append(
        kt.knn_predict(X_train=X, X_test=test_point.reshape(1, -1), y_train=Y, k=3, p=1)[0])

t4 = time.perf_counter()#Brute force search end time and kd tree search start time

for test_point in X_test:
    label, node_list = kd_tree_all.knn_face_recognition(values=test_point, k=3,sigma=1)
    y_hat_test_kd.append(label)

t5 = time.perf_counter()#kd tree search end time

count = 0
count_kd = 0
# Calculation Accuracy
for i in range(len(Y_test)):
    if Y_test[i] == y_hat_test[i]:
        count = count + 1
    if Y_test[i] == y_hat_test_kd[i]:
        count_kd += 1

print(len(Y_test))
print(len(Y))


print('Time consuming to build kd tree:{:.4f}s'.format(t2 - t1))

print('knn algorithm without kd tree time consuming:{:.4f}s'.format(t4 - t3))

print('knn algorithm with kd tree time consuming:{:.4f}s'.format(t5 - t4))

print("Accuracy without kd tree: {:.4f}".format((count / len(Y_test))*100))

print("Accuracy with kd tree: {:.4f}".format((count_kd / len(Y_test))*100))


