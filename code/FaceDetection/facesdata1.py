
import os
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib.pyplot as plt
import face_recognition as fr
import numpy as np
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

            if Y[i] != Y[j]:
                if len(dis_diff)>112256:
                    continue
                count_diff +=1
                dis_diff.append(X[i][d] - X[j][d])

            if  Y[i] == Y[j]:
                count_sam += 1
                dis_sam.append(X[i][d] - X[j][d])





plt.hist(x=dis_diff,bins=1000, normed=1, facecolor="blue", edgecolor="black", alpha = 0.7)

#plt.xlabel('# of Nearest Neighbors (k)')
plt.xlabel('#Interval')
plt.ylabel('frequecy')

plt.show()

plt.hist(x=dis_sam,bins=1000, normed=1, facecolor="blue", edgecolor="black", alpha = 0.7)
#plt.xlabel('# of Nearest Neighbors (k)')
plt.xlabel('#Interval')
plt.ylabel('frequecy')

plt.show()


print("The variance of the difference of each dimension of face feature vectors of different person: ",np.var(dis_diff))
print("The variance of the difference of each dimension of face feature vectors of same person: ",np.var(dis_sam))

print("Group of the difference person's face: ",count_diff)
print("Group of the same person's face: ",count_sam)
