import os
import dlib
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition as fr
from face_recognition.face_recognition_cli import image_files_in_folder

#Define a function for training the model
def train(train_dir,model_save_path='face_recog_model.clf',n_neighbors=3,kclf=None,knn_algo='ball tree'):
    #Generating training sets
    print("Training KNN classifier ~~~~")
    X=[]
    Y=[]
    #遍历训练集中的每一个人  class_dir是获取到的人名
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue #结束当前循环，进入下一个循环

        #遍历这个人每一张图片 img_path是获取到的具体的人文件夹下的一张图片名
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            try:
                image = fr.load_image_file(img_path)
                boxes = fr.face_locations(image)
           #对于当前的图片，增加编码至训练集
                #print(img_path)
                X.append(fr.face_encodings(image,known_face_locations=boxes)[0])  #返回128维度的向量
                Y.append(class_dir)
            except:
                print('failed!')

    #决定K值 (KNN)
    if n_neighbors is None:
        n_neighbors=3

    #创建并且训练分类器
    #if model_save_path is None:
    #   knn_clf=neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

        #else:
        # with open(model_save_path) as f:
    #       knn_clf=pickle.load(f)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X, Y)

    #保存训练好的分类器
    if model_save_path is not None:
        with open(model_save_path,'wb') as f:
            pickle.dump(knn_clf,f)
    print("Training completed!")
    return  knn_clf


#使用模型预测
def predict(img_path,knn_clf=None,model_path=None,distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise  Exception("KNN Classifier is necessary! You can use knn_clf or model_path.")

    #加载训练好的KNN模型
    if knn_clf is None:
        with open(model_path,"rb") as f:
            knn_clf=pickle.load(f)

    #加载图片，发现人脸的位置
    X_img=fr.load_image_file(img_path)
    X_face_locations=fr.face_locations(X_img)

    #对测试图片中的人脸编码  返回的是128个人脸特征构成的向量
    print("I found {} face(s) in this photograph.".format(len(X_face_locations)))
    preds=[]

    for i in range(len(X_face_locations)):
        top, right, bottom, left = X_face_locations[i]
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        encondings = fr.face_encodings(X_img, known_face_locations=X_face_locations)[i].reshape(1, -1)
        closest_distances = knn_clf.kneighbors(encondings, n_neighbors=1)
        are_matches = [closest_distances[0][0][0] <= distance_threshold]

        for pred, rec in zip(knn_clf.predict(encondings), are_matches):

            loc = X_face_locations[i]
            if rec:
                preds.append([pred, loc])
            else:
                preds.append(["unknown", loc])

    #利用KNN model找出与测试人脸最匹配的人脸



    #预测类别
    return  preds


#人脸识别可视化函数
def show_names_on_image(img_path, predictions):
    pil_image=Image.open(img_path).convert("RGB")
    draw=ImageDraw.Draw(pil_image)

    for name,(top,right,bottom,left) in predictions:
        #用Pillow模块画出人脸边界盒子
        draw.rectangle(((left,top),(right,bottom)),outline=(255,0,255))

        #pillow里可能生成非UTF-8格式，所以做如下的转换
        name=name.encode("UTF-8")
        name=name.decode("ascii")

        #在人脸下写下名字，作为标签
        text_width,text_height=draw.textsize(name)
        draw.rectangle(((left,bottom-text_height-10),(right,bottom)),
                       fill=(255,0,255),outline=(255,0,255))
        draw.text((left+6,bottom-text_height-5),name,fill=(255,255,255))

    del draw
    pil_image.show()
    dlib.hit_enter_to_continue()




def knn_face_recognition(model_path=None, test_path=None, train_path=None):

    try:

        if train_path is not None:

            train(train_path,model_save_path=model_path, _neighbors=2)

        else:
            print("The path of the training set is necessary unless there exists an already trained knn model.")

        if test_path is not None:

            for image_file in os.listdir(test_path):
                full_file_path = os.path.join(test_path, image_file)
                print("Finding faces in {} ...".format(image_file))

            # 利用分类器，找出所有的人脸：
            # 要么传递一个classifier文件名，要么传一个classifier模型实例
                predictions = predict(full_file_path, model_path=model_path)

            # 打印结果

                for name, (top, right, bottom, left) in predictions:
                    print("Found {}，Location:({},{},{},{})".format(name, top, right, bottom, left))

            # 在图片上显示预测结果
                show_names_on_image(os.path.join(test_path, image_file), predictions)

        else:
            print("The path of the testing set is necessary !!!")
    except:

        print("There is something wrong with your path, please check if the path is correct and if the knn model exists.")
        print("Usage: knn_face_recognition(model_path, test_path, train_path)")


