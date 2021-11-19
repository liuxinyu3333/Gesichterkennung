
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import os

from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
#import matplotlib.pyplot as plt
import face_recognition as fr
from PIL import Image, ImageDraw

class Node(object):

    def __init__(self, values=None, label=None, split=None, parent=None, left_child=None, right_child=None):

        self.values = values # Node Value

        self.label = label # Node labels

        self.split = split # Performing segmentation of dimensions

        self.parent = parent # Parent Node

        self.left_child = left_child # Left sub-tree

        self.right_child = right_child # Right sub-tree


class Kd_Tree(object):

    def __init__(self, values, labels):

        self.__length = 0  #Number of kd tree nodes

        self.root = self.__create(values, labels)  # The nodes of the kd tree

    def __create(self, values, labels, parent = None):  # Building trees
        # m is the number of rows (number of samples),
        # n is the number of columns (number of features)
        m, n = values.shape
        labels = labels.reshape(m, 1)
        if m == 0:
            return None  # Empty (sub)trees
        # Calculate the variance of each column
        var_list = [np.var(values[:, column]) for column in range(n)]
        # Find the index of the column with the largest variance as the dimension of the split
        pos = var_list.index(max(var_list))
        # Obtain a index sequence that ranks the column's feature values in ascending order
        index_list = values[:, pos].argsort()
        median = index_list[m//2]  # Find the index of the median

        if m ==1:
            self.__length += 1
            # Returns itself when the number of samples is 1
            return Node(values=values[median], label=labels[median], split=pos,
                        parent=parent, left_child=None, right_child=None)
        # Generate a node
        node = Node(values=values[median], label=labels[median], split=pos,
                        parent=parent, left_child=None, right_child=None)

        #  Building ordered subtrees
        left_tree_values = values[index_list[:m//2]]  # All values of the left subtree
        left_tree_labels = labels[index_list[:m//2]]  # All labels of the left subtree

        left_child = self.__create(left_tree_values, left_tree_labels, node)
        if m == 2:
            right_child = None  #Only left subtree, no right subtree

        else:
            right_tree_values = values[index_list[m//2+1:]]  # All values of the right subtree
            right_tree_labels = labels[index_list[m//2+1:]]  # All labels of the right subtree
            right_child = self.__create(right_tree_values, right_tree_labels, node)

        # The left and right subtrees recursively call themselves,
        # returning the root node of the subtree
        node.left_child = left_child
        node.right_child = right_child

        self.__length += 1

        return node


    def similarity(self, a, b, sigma=9.0):
        # Store the number of dimensions
        dim = len(a)

        # Set initial weight to 0
        weight = 0
        # Calculate similarity
        for d in range(dim):
            # Weight the difference in each dimension with Gaussian function
            # and add up the weights of all dimensions
            weight += np.exp(-abs(a[d] - b[d]) ** 2 / (2 * sigma ** 2))
        # Get overall geometric similarity
        sim = weight / dim

        return sim


    def Gaussian(self, distance, sigma=9.0):
        # Input a distance and return it`s weight
        weight = np.exp(-distance ** 2 / (2 * sigma ** 2))
        return weight

    def minkowski_distance(self, a, b, p=2):
        # Store the number of dimensions
        dim = len(a)

        # Set initial distance to 0
        distance = 0

        # Calculate minkowski distance using parameter p
        for d in range(dim):
            distance += abs(a[d] - b[d]) ** p

        distance = distance ** (1 / p)

        return distance

    def find_nearest_neighbour(self, values):  # Find the closest point to the sample point in the kd tree

        values = np.array(values)
        if self.__length == 0:
            return None  # Empty trees

        node = self.root  # Starting from the root node
        if self.__length == 1:
            return node  # A kd tree with only one node

        while True:
            cur_split = node.split  # Dimensions for division during tree building
            # The value of the sample to be judged in this dimension
            # is less than the value of the current node in this dimension
            if values[cur_split] < node.values[cur_split]:
                # If the left subtree is empty,
                # the current node is returned
                if node.left_child == None:
                    return node

                # If the left subtree is not empty,
                # go to the left subtree and keep looking
                node = node.left_child

            # The value of the sample to be judged in this dimension is greater than
            # the value of the current node in this dimension
            else:
                if node.right_child == None:
                    # If the right subtree is empty,
                    # the current node is returned
                    return node

                # If the right subtree is not empty,
                # go to the right subtree and continue searching
                node = node.right_child

    def knn_face_recognition(self, values, k, sigma = 9.0):

        values = np.array(values)
        node = self.find_nearest_neighbour(values)  # Find the node closest to the sample to be judged
        if node == None:
            return None  # Empty trees

        # used to store the information of the k closest points
        # to the sample to be judged, which will be updated continuously
        node_list = []
        distance = self.minkowski_distance(values, node.values) # Distance between test point and nearest point
        neigh_dis = distance  # Update the farthest distance from the sample to be judged in the current node_list

        # Store the distance between the selected node and the sample to be judged,
        # the value of the node and the label information into node_list
        node_list.append([distance, tuple(node.values), node.label[0]])

        while True: # Keep returning to the previous node
            if node == self.root:
                break  #If you return to the root node, the loop ends

            parent = node.parent
            par_dis = self.minkowski_distance(values, parent.values)

            # Calculate the distance between the parent node and the test point,
            # and compare it with the maximum distance in the node list
            if k > len(node_list) or par_dis < neigh_dis:
                #The length of the node list is less than the value of k
                # or the new distance is less than the maximum distance
                node_list.append([par_dis, tuple(parent.values), parent.label[0]])
                node_list.sort()
                # Update the maximum distance
                neigh_dis = node_list[-1][0] if k >= len(node_list) else node_list[k - 1][0]

            # Determine the size of the distance between the sample to be judged and
            # the dividing line of the current node and the maximum distance in the current node list
            if k > len(node_list) or abs(values[parent.split] - parent.values[parent.split]) < neigh_dis:#The distance from the point to be measured to the upper segmentation plane
                # The current node has other branches
                other_child = parent.left_child if parent.left_child != node else parent.right_child
                if other_child != None:
                    if values[parent.split] - parent.values[parent.split] <= 0:
                        # The test point is on the left side of the hyperplane of the sub-node
                        self.left_search(values, other_child, node_list, k)
                    else:
                        # The test point is on the right side of the hyperplane of the sub-node
                        self.right_search(values, other_child, node_list, k)
            #return to the previous level
            node = parent

        label_dict = {}
        label_weight = {}
        node_list = node_list[:k]
        # Store the number of occurrences of each label and the weight of each label in the dictionary
        for n in node_list:
            label_weight[n[2]] = label_weight.get(n[2], 0) + self.similarity(a=values, b=n[1], sigma=sigma)
            label_dict[n[2]] = label_dict.get(n[2], 0) + 1

        for k,v in label_weight.items():
            label_weight[k] = v/label_dict.get(k,1)

        # Sort labels in descending order of weight
        sorted_label = sorted(label_weight.items(), key=lambda x: x[1], reverse=True)

        # Return a list of the label with the largest weight and the k nodes closest to the sample to be judged
        return sorted_label[0][0]


    def left_search(self, values, node, nodeList, k):

        # Update nodelist data before each comparison
        nodeList.sort()  # Sorting the list of nodes by distance
        neigh_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]

        if node.left_child == None and node.right_child == None:  # leaf node
            dis = self.minkowski_distance(values, node.values)
            if k > len(nodeList) or dis < neigh_dis:
                nodeList.append([dis, tuple(node.values), node.label[0]])
                nodeList.sort()  # Sorting the list of nodes by distance
            return

        if node.left_child != None:

            self.left_search(values, node.left_child, nodeList, k)

        dis = self.minkowski_distance(values, node.values)
        if k > len(nodeList) or dis < neigh_dis:
            nodeList.append([dis, tuple(node.values), node.label[0]])
            nodeList.sort()  # Sorting the list of nodes by distance
        # Right sub-tree
        if k > len(nodeList) or abs(values[node.split] - node.values[node.split]) < neigh_dis:  # Need to search the right subtree
            if node.right_child != None:
                self.right_search(values, node.right_child, nodeList, k)

        return nodeList


    def right_search(self, values, node, nodeList, k):

        nodeList.sort()  # Sorting the list of nodes by distance
        neigh_dis = nodeList[-1][0] if k >= len(nodeList) else nodeList[k - 1][0]
        if node.left_child == None and node.right_child == None:  # leaf node
            dis = np.sqrt(sum((values - node.values) ** 2))
            if k > len(nodeList) or dis < neigh_dis:
                nodeList.append([dis, tuple(node.values), node.label[0]])
                nodeList.sort()  # Sorting the list of nodes by distance
            return

        if node.right_child != None:
            self.right_search(values, node.right_child, nodeList, k)

        dis = self.minkowski_distance(values, node.values)
        if k > len(nodeList) or dis < neigh_dis:
            nodeList.append([dis, tuple(node.values), node.label[0]])
            nodeList.sort()  # Sorting the list of nodes by distance
        # Left sub-tree
        if k > len(nodeList) or abs(values[node.split] - node.values[node.split]) < neigh_dis:  # Need to search the left subtree
            self.left_search(values, node.left_child, nodeList, k)

        return nodeList

def test():
    print("Training KNN classifier ~~~~")
    X = []
    Y = []
    # Iterate through each person in the training set. "class_dir" is the name of the person obtained
    for class_dir in os.listdir("examples/train"):
        if not os.path.isdir(os.path.join("examples/train", class_dir)):
            continue

        # Iterate through each image of the person, img_path is the name of an image in the specific person's folder that was obtained
        for img_path in image_files_in_folder(os.path.join("examples/train", class_dir)):
            try:
                image = fr.load_image_file(img_path)
                boxes = fr.face_locations(image)

                # print(img_path)
                X.append(fr.face_encodings(image, known_face_locations=boxes)[0])  # Returns a vector of 128 dimensions
                Y.append(class_dir)

            except:
                print('failed!')

    dataArray = np.array(X)

    labelArray = np.array(Y)

    print("Training finished !!!!!")

    t1 = time.perf_counter()
    print("Building kd tree ~~~~~~~~")
    kd_tree_all = Kd_Tree(dataArray, labelArray)
    print("Build finish!!!!!!!")
    # kd_list = kd_tree.transfer_list(kd_tree.root)

    t2 = time.perf_counter()


    for image_file in os.listdir("examples/test"):
        full_file_path = os.path.join("examples/test", image_file)
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
            encondings = fr.face_encodings(X_img, known_face_locations=X_face_locations)[i]

            label, node_list = kd_tree_all.knn_face_recognition(encondings, k=3)

            if node_list[0][0] < 0.5 :
                preds.append([label, loc])
            else:
                preds.append(['unkown', loc])


        for pre in preds:
            print('Labels for prediction points:%s' % pre)


    t3 = time.perf_counter()



    print('Time consuming to build kd tree:{:.4f}s'.format(t2 - t1))

    print('knn algorithm time consuming:{:.4f}s'.format(t3 - t2))

    X_train, X_test, y_train, y_test = train_test_split(dataArray, labelArray, test_size=0.2,
                                                        random_state=1)

    kd_tree_al = Kd_Tree(X_train, y_train)
    y_hat_test = []
    for data in X_test:
        label, node_list = kd_tree_al.knn_face_recognition(data, k=3)
        y_hat_test.append(label)

    print(accuracy_score(y_test, y_hat_test))

