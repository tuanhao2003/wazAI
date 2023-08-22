import os
import numpy
import tensorflow
import tensorflow.keras.layers as layers
import cv2
from sklearn.model_selection import train_test_split

# khai báo thư mục chứa dataset và danh sách nhãn
dataFolder = "F:/Compositions/PetImages"
allLabels=["Cat", "Dog"]

#mảng chứa dữ liệu và nhãn
Datas=[]
Labels=[]

# duyệt qua folder chứa dataset
for classifyFolders in os.listdir(dataFolder):
# kiểm tra xem 2 thư mục trong thư mục petimages có tên giống với nhãn nào ở trên ko
    if(classifyFolders in allLabels):
# lấy label là index của mảng label ở trên
        labelIndex=allLabels.index(classifyFolders)
# tạo ra đường dẫn tới thư mục chứa hình ảnh
        classifyPath=os.path.join(dataFolder, classifyFolders)
        for imgFile in os.listdir(classifyPath):
# quét qua từng ảnh bằng cv2 với chế độ ảnh xám
            imgPath=os.path.join(classifyPath, imgFile)
            img=cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
# thêm ảnh vào mảng data và label vào mảng label
            Datas.append(img)
            Labels.append(labelIndex)

# đổi mảng về kiểu mảng numpy đồng thời đổi size hình trong mảng data
Datas=numpy.array(Datas).reshape(-1, 28, 28, 1)
Labels=numpy.array(Labels)

# chia mảng data và label thành 2phần train và test với 20% là test
trainDatas, testDatas, trainLabels, testLabels= train_test_split(Datas, Labels, test_size=0.2)

# xây dựng mô hình train
model=tensorflow.keras.Sequential(
    [
        layers.Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax")
    ]
)

# biên dịch và train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(trainDatas, trainLabels, epochs=5)
model.evaluate(testDatas, testLabels)