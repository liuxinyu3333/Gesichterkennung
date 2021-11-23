
import os
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib.pyplot as plt
import face_recognition as fr

print("Training KNN classifier ~~~~")
X = []
Y = []
# Iterate through each person in the training set. "class_dir" is the name of the person obtained
for class_dir in os.listdir("examples/Harrypotter_test"):#Traverse the folders under this folder
    if not os.path.isdir(os.path.join("examples/Harrypotter_test", class_dir)):#Path does not exist
        continue

    # Iterate through each image of the person,
    # img_path is the name of an image in the specific person's folder that was obtained
    for img_path in image_files_in_folder(os.path.join("examples/Harrypotter_test", class_dir)):
        try:
            image = fr.load_image_file(img_path)#Load image from image path
            boxes = fr.face_locations(image)#Get the position of the face in the picture
            # Returns a vector of 128 dimensions
            X.append(fr.face_encodings(image, known_face_locations=boxes)[0])#Training set face vector
            Y.append(class_dir)#Training set face vector corresponding label
            print("label: ", class_dir)
            print(fr.face_encodings(image, known_face_locations=boxes)[0])
        except:
            print('failed!')
print("Training finished !!!!!")

dis_diff=[]
dis_sam=[]
count_sam = 0
count_diff = 0
max_diff = 0
for i in range(len(X)-1):
    dim = len(X[i])
    for j in range(i+1,len(X)):
        print("The difference of feature value of ", Y[i], ",", Y[j])
    # Set initial distance to 0
        distance_sam = 0
        distance_diff = 0
        max_sam = 0
    # Calculate minkowski distance using parameter p
        for d in range(dim):

        #if abs(X[i][d] - X[i+1][d])>0.08:
            if abs(X[i][d] - X[j][d]) >= max_diff and Y[i] != Y[j]:
                max_diff = abs(X[i][d] - X[j][d])
            if Y[i] != Y[j]:
                distance_diff += abs(X[i][d] - X[j][d])

            if abs(X[i][d] - X[j][d]) >= max_sam and Y[i] == Y[j]:
                max_sam = abs(X[i][d] - X[j][d])

            if  Y[i] == Y[j]:
                distance_sam += abs(X[i][d] - X[j][d])

        if Y[i] == Y[j]:
            count_sam += 1
            print("the diffrerntest feature value:", max_sam )
            dis_sam.append(distance_sam / dim)
            print("same average:", distance_sam/dim)
        else:
            count_diff += 1
            dis_diff.append(distance_diff/dim)


plt.plot(range(1,count_diff+1 ),dis_diff)
#plt.xlabel('# of Nearest Neighbors (k)')
plt.xlabel('# of Weighted knn (sigma)')
plt.ylabel('Accuracy (%)')

plt.show()

plt.plot(range(1,count_sam+1 ),dis_sam)
#plt.xlabel('# of Nearest Neighbors (k)')
plt.xlabel('# of Weighted knn (sigma)')
plt.ylabel('Accuracy (%)')

plt.show()





print("The largest dimensionality difference is:",max_diff)