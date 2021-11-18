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
        except:
            print('failed!')


accuracies = []
y_hat_test = []#Prediction results of knn algorithm without kd tree construction

#Testing the effect of different k, p, and sigma on the accuracy of knn
for k in range(1,len(Y)+1):
    y_hat_test = []
    count = 0  # right result of knn
    im = 0
    for test_point in X_test:#Traverse all faces to be predicted
        #Predict face identity by weighted knn
        label = kt.knn_predict(X_train=X, X_test=test_point.reshape(1, -1), y_train=Y, k=k, p=1)
        if len(label) > 1:
            print("many result:", im)
            for la in label:
                if la == Y_test[im]:
                    count = count + 1
                    break
        else:
            print("one result:", im)
        #Forecast result storage
            if label[0] == Y_test[im]:
                count = count + 1
        im += 1

    # Calculation Accuracy

    print("Accuracy : {:.2f}%, bei  k : ".format((count / len(Y_test)) * 100), format(k))
    accuracies.append(count / len(Y_test))

# Plot the results

plt.plot(range(1,len(Y)+1), accuracies)
plt.xlabel('# of Nearest Neighbors (k)')
#plt.xlabel('# of minkowski_distance (p)')
#plt.xlabel('# of Weighted knn (sigma)')
plt.ylabel('Accuracy (%)')
plt.show()
