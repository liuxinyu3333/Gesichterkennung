
import KNN_real as kt

import os
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib.pyplot as plt
import face_recognition as fr

print("Training KNN classifier ~~~~")
X = []
Y = []
# Iterate through each person in the training set. "class_dir" is the name of the person obtained
for class_dir in os.listdir("examples/Harrypotter"):#Traverse the folders under this folder
    if not os.path.isdir(os.path.join("examples/Harrypotter", class_dir)):#Path does not exist
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/Harrypotter", class_dir)):
        try:
            image = fr.load_image_file(img_path)#Load image from image path
            boxes = fr.face_locations(image)#Get the position of the face in the picture
            # Returns a vector of 128 dimensions
            X.append(fr.face_encodings(image, known_face_locations=boxes)[0])#Training set face vector
            Y.append(class_dir)#Training set face vector corresponding label
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
            image = fr.load_image_file(img_path)#Load image from image path
            boxes = fr.face_locations(image)#Get the position of the face in the picture

            # Returns a vector of 128 dimensions
            X_test.append(fr.face_encodings(image, known_face_locations=boxes)[0])#Test set face vector
            Y_test.append(class_dir)#Test set face vector corresponding label

            #print(img_path)

        except:
            print('failed!')

#accuracies_weight = []
accuracies = []
y_hat_test = []#Prediction results of knn algorithm

#Testing the effect of different k on the accuracy of knn

for k in range(1,len(Y)+1):
    y_hat_test = []
    i = 0
    count = 0  # right result of knn
    for test_point in X_test:#Traverse all faces to be predicted
        #Predict face identity by weighted knn
        label = kt.weighted_knn_predict(X_train=X, X_test=test_point.reshape(1, -1), y_train=Y, k=k, sigma = 0.4, threshold = 0.9)[0]
        if Y_test[i] == label:
            count = count + 1
        else:
            print("i=",i)
            print(Y_test[i])
            print(label)
        #Forecast result storage
        i=i+1

    # Calculation Accuracy

    print("Accuracy : {:.2f} %, bei k : ".format((count / len(Y_test)) * 100), format(k))
    accuracies.append(count / len(Y_test))


# Plot the results

plt.plot(range(1,len(Y)+1), accuracies,label = "knn")
plt.xlabel('# of Nearest Neighbors (k)')

plt.ylabel('Accuracy (%)')


plt.show()
