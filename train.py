import os
import numpy
import tensorflow
import tensorflow.keras.layers as layers
import cv2
from sklearn.model_selection import train_test_split

# khai báo thư mục chứa dataset và danh sách nhãn
# dataFolder = "F:/Compositions/PetImages"
dataFolder="C:/Users/Admin/Downloads/meme"
allLabels=["cheem", "pepe", "popcat"]

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
model=tensorflow.keras.Sequential(
    [
# tích chập
        layers.Conv2D(32,(3,3), activation="relu", input_shape=(28,28,1)),
# gộp pixel
        layers.MaxPooling2D((2,2)),
# tích chập
        layers.Conv2D(64,(3,3), activation="relu", padding="same"),
# gộp pixel
        layers.MaxPooling2D((2,2)),
# tích chập
        layers.Conv2D(128,(3,3), activation="relu", padding="valid"),
# gộp pixel
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
# mạng nơron đợt 1
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
# mạng nơron đợt 2 (đầu ra label)
        layers.Dense(3, activation="softmax")
    ]
)

# biên dịch và train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(trainDatas, trainLabels, epochs=200)
_, accuRate= model.evaluate(testDatas, testLabels)
print(accuRate*100,"%")
model.save("memeDetect.keras")