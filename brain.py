import os
import numpy
import tensorflow
from tensorflow.keras.layers import Dense, Reshape, Dropout, Flatten, Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split

def trainClassify(dataFolder, allLabels, loop):
#mảng chứa dữ liệu và nhãn
    Datas=[]
    Labels=[]

# duyệt qua folder chứa dataset
    for classifyFolders in os.listdir(dataFolder):
# lấy label là index của mảng label ở trên
        labelIndex=allLabels.index(classifyFolders)
# tạo ra đường dẫn tới thư mục chứa hình ảnh
        classifyPath=os.path.join(dataFolder, classifyFolders)
        for imgFile in os.listdir(classifyPath):
# quét qua từng ảnh bằng cv2 với chế độ ảnh xám
            imgPath=os.path.join(classifyPath, imgFile)
            img=cv2.imread(imgPath, 0)
# check lỗi đặt tên hình
            if img is None:
                print('Wrong path:', imgPath)
            else:
                img=cv2.resize(img, (28,28))
# thêm ảnh vào mảng data và label vào mảng label
                Datas.append(img)
                Labels.append(labelIndex)

# đổi mảng về kiểu mảng numpy đồng thời đổi size hình trong mảng data
    Datas=numpy.array(Datas).reshape(-1,28,28,1)
    Labels=numpy.array(Labels)

# chia mảng data và label thành 2phần train và test với 20% là test
    trainDatas, testDatas, trainLabels, testLabels= train_test_split(Datas, Labels, test_size=0.2)

# xây dựng mô hình train
    model=tensorflow.keras.Sequential([
# tích chập
        Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)),
# gộp pixel
        MaxPooling2D((2,2)),
# tích chập
        Conv2D(64,(3,3), activation="relu", padding="same"),
# gộp pixel
        MaxPooling2D((2,2)),
# tích chập
        Conv2D(128,(3,3), activation="relu", padding="valid"),
# gộp pixel
        MaxPooling2D((2,2)),
        Flatten(),
# mạng nơron đợt 1
        Dense(32, activation="relu"),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(256, activation="relu"),
# mạng nơron đợt 2 (đầu ra label), ########số lớp nơ ron bằng số labels###########
        Dense(len(allLabels), activation="softmax")
    ])

# biên dịch và train
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(trainDatas, trainLabels, epochs=loop)
    _, accuRate= model.evaluate(testDatas, testLabels)

# in cấu trúc mô hình
#    model.summary()

# in tỉ lệ nhận dạng đúng
    print(accuRate*100,"%")
    model.save("classify.keras")

##################################################################################################################################

# tạo mô hình gen
# đầu vào là vector ngẫu nhiên
# 3 lớp nơ ron
# đầu ra là hình 64x64 màu (3 giá trị gồm dài - rộng ảnh, số kênh màu)
def init_gener(imgShape):
    model=tensorflow.keras.Sequential([
        Dense(128, activation="relu", input_shape=imgShape),
        Dense(256, activation="relu"),
        Dense(512, activation="relu"),
        Dense(784, activation="sigmoid"),
        Reshape((64,64,3))
    ])
    return model

# tạo mô hình kiểm tra ảnh thật giả
# đầu vào là hình 64x64 màu 
# 3 lớp nơron đối lập với mô hình gen 
# đầu ra là tỉ lệ giả mạo(1 giá trị)
def init_checker(shape):
    model=tensorflow.keras.Sequential([
        Flatten(input_shape=shape),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    return model


# tạo mô hình GAN
def init_creator(gener, checker):
    model= tensorflow.keras.Sequential([
        gener,
        checker
    ])
    return model

# hàm mất mát cho creator
def creator_loss(fake_output):
    return tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)(tensorflow.ones_like(fake_output), fake_output, from_logits=True)

# train creator
def trainCreator(dsImgs, loop, datPerloop):
# khởi tạo
    gener=init_gener(12288)
    checker=init_checker(12288)
    creator=init_creator(gener, checker)

# biên dịch checker và creator
    checker.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    creator.compile(optimizer="adam", loss=creator_loss)

    realImgindex=0
    for trainTime in range(loop):
        for batch in range(datPerloop):
            realImg=[]
# train checker
            imgList=os.listdir(dsImgs)[realImgindex, realImgindex+datPerloop]
            for i in imgList:
                realImg.append(cv2.resize( cv2.imread(dsImgs+"/"+i),(64,64)))
            realImgindex+=datPerloop
            realImg=numpy.array(realImg)/255.0
            fakeImg=gener.predict(numpy.random.rand(datPerloop, 12288))

            realLabel=numpy.ones((datPerloop, 1))
            fakeLabel=numpy.zeros((datPerloop, 1))

            realPercent=checker.train_on_batch(realImg, realLabel)
            fakePercent=checker.train_on_batch(fakeImg, fakeLabel)

# train gener
            randVec=numpy.random.rand(datPerloop, 12288)
            genLabel=numpy.ones((datPerloop, 1))

            creatorLoss=creator.train_on_batch((randVec, genLabel))

# in thong tin
            print("epoch:", trainTime, "- tỉ lệ ảnh thật:", realPercent,"- tỉ lệ ảnh giả:", fakePercent, "- độ fake của creator:", creatorLoss)
        gener.save_weights("gen.h5")

##################################################################################################################################

def create_img(inputImgLink, datasetLink):
    gener=init_gener(12288)
    gener.load_weights("gen.h5")
    for data in os.listdir(datasetLink):
        datasetImg=[]
        datasetImg.append(cv2.imread(datasetLink+"/"+data))
# Chọn một hình ngẫu nhiên từ tập dataset_images
    getDataimg = numpy.random.choice(datasetImg)

# Biến đổi đặc điểm của hình inputImgLink thành hình ảnh mới
# chuẩn hóa hình ảnh
    inputImg=cv2.resize(cv2.imread(inputImgLink), (64,64)).reshape(1,-1)
    imgCreated = gener.predict(inputImg)+getDataimg

    cv2.imwrite("newImage.jpg", imgCreated)