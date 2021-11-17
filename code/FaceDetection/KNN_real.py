

# Initial imports
#
import time
import dlib


import os
from face_recognition.face_recognition_cli import image_files_in_folder
import pandas as pd
import numpy as np
import face_recognition as fr

from PIL import Image, ImageDraw

def Gaussian(distance, sigma = 9.0):
    #Input a distance and return it`s weight
    weight = np.exp(-distance**2/(2*sigma**2))
    return weight

def minkowski_distance(a, b, p=1):
    # Store the number of dimensions
    dim = len(a)

    # Set initial distance to 0
    distance = 0

    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p

    distance = distance ** (1 / p)

    return distance


def knn_predict(X_train, X_test, y_train, k, p = 2):
    # Counter to help with label voting
    # from collections import Counter

    # Make predictions on the test data
    # Need output of 1 prediction per test data point

    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)


        # Store distances in a dataframe
        # print(distances)
        df_dists= pd.DataFrame(data=distances, columns=['dist'])
        #df_dists['labels'] = y_train
        #print(df_dists)
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]#dataframe (Sort and retain only the first k items)
        df_in = df_nn.to_dict(orient='split').get('index')#dictionary of index
        #print(df_in)
        df_sort = df_nn.to_dict(orient='list').get('dist')#dictionary of distances
        # print(df_sort)
        df_index = []
        for index in df_in:
            df_index.append(y_train[index])#list of label(Condidate)
        #print(df_index)
        counts = {}

        for i in range(k):
            voteIlabel = df_index[i]
            counts[voteIlabel] = counts.get(voteIlabel,0) + 1


        sortedClassCount = sorted(counts.items(), key=lambda x:x[1], reverse=True)

        # Takes the first distance value and determines if it is less than the threshold value
        dis = df_sort[0]
        #print(dis)
        if(dis <= 5):

            prediction = sortedClassCount[0][0]

        else:
            prediction = 'unknow'

        # Append prediction to output list
        y_hat_test.append(prediction)
        #print(y_hat_test)

    return y_hat_test


def weighted_knn_predict(X_train, X_test, y_train, k, p = 2, sigma = 9.0):
    # Counter to help with label voting
    # from collections import Counter

    # Make predictions on the test data
    # Need output of 1 prediction per test data point

    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)


        # Store distances in a dataframe
        # print(distances)
        df_dists= pd.DataFrame(data=distances, columns=['dist'])
        #df_dists['labels'] = y_train
        #print(df_dists)
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]  # dataframe (Sort and retain only the first k items)
        df_in = df_nn.to_dict(orient='split').get('index')  # dictionary of index
        # print(df_in)
        df_sort = df_nn.to_dict(orient='list').get('dist')  # dictionary of distances
       # print(df_sort)
        df_index = []
        for index in df_in:
            df_index.append(y_train[index])
        #print(df_index)
        counts = {}
        classCount = {}
        for i in range(k):
            voteIlabel = df_index[i]
            # Gewichtung
            weight = Gaussian(df_sort[i], sigma=sigma)
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + weight
            counts[voteIlabel] = counts.get(voteIlabel,0) + 1

        for k,v in classCount.items():
            classCount[k] = v/counts.get(k,1)

        sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)

        # Takes the first distance value and determines if it is less than the threshold value
        dis = df_sort[0]
        #print(dis)
        if(dis <= 5):

            prediction = sortedClassCount[0][0]

        else:
            prediction = 'unknow'

        # Append prediction to output list
        y_hat_test.append(prediction)
        #print(y_hat_test)

    return y_hat_test


def knn_face_recognition(train_path,test_path):

    print("Training KNN classifier ~~~~")
    X=[]
    Y=[]
    #Iterate through each person in the training set.
    #"class_dir" is the name of the person obtained
    for class_dir in os.listdir(train_path):
        if not os.path.isdir(os.path.join(train_path,class_dir)):
            continue

        #Iterate through each image of the person,
        #img_path is the name of an image in the specific person's folder that was obtained
        for img_path in image_files_in_folder(os.path.join(train_path,class_dir)):
            try:
                image = fr.load_image_file(img_path)
                boxes = fr.face_locations(image)
                print(img_path)

                # Returns a vector of 128 dimensions
                X.append(fr.face_encodings(image,known_face_locations=boxes)[0])
                Y.append(class_dir)

            except:
                print('failed!')

    print("Training finished !!!!!")
    t1 = time.perf_counter()

    for image_file in os.listdir(test_path):
        full_file_path = os.path.join(test_path, image_file)
        print("Finding faces in {} ...".format(image_file))

        X_img = fr.load_image_file(full_file_path)
        X_face_locations = fr.face_locations(X_img)

    # Encoding the faces in the test images, returns a vector of 128 face features
        print("I found {} face(s) in this photograph.".format(len(X_face_locations)))
        preds = []

        for i in range(len(X_face_locations)):
            top, right, bottom, left = X_face_locations[i]
            loc = X_face_locations[i]
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))
        #Face encoding of test images
            encondings = fr.face_encodings(X_img, known_face_locations=X_face_locations)[i].reshape(1, -1)

            predicts=knn_predict(X_train=X, X_test=encondings, y_train=Y, k=3, p=2, sigma=9.0)

            preds.append([predicts, loc])

        #show_names_on_image(os.path.join(test_path, image_file), preds)\
        for pre in preds:
            print('Labels for prediction points:%s' % pre)

    t2 = time.perf_counter()
    print('knn algorithm time consuming:{:.4f}s'.format(t2 - t1))

    return  preds

def show_names_on_image(img_path, predictions):
    pil_image=Image.open(img_path).convert("RGB")
    draw=ImageDraw.Draw(pil_image)

    for name,(top,right,bottom,left) in predictions:
        #Draw the face bounding box with the Pillow module
        draw.rectangle(((left,top),(right,bottom)),outline=(255,0,255))

        #pillow may generate non-UTF-8 format, so do the following conversion
        name=name[0].encode("UTF-8")
        name=name.decode("ascii")

        #Write the name under the face as a tag
        text_width,text_height=draw.textsize(name)
        draw.rectangle(((left,bottom-text_height-10),(right,bottom)),
                       fill=(255,0,255),outline=(255,0,255))
        draw.text((left+6,bottom-text_height-5),name,fill=(255,255,255))

    del draw
    pil_image.show()
    dlib.hit_enter_to_continue()


