import os
import numpy
import cv2
from tensorflow.keras.models import load_model
from brain import trainClassify, create_img

# tiền xử lý ảnh đầu vào(chuẩn hóa)
def preImg(imgPath):
    img = cv2.imread(imgPath, 0)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
# đưa điểm sáng về giá trị 0-1
    img = img.astype('float32') / 255.0
    return img

# nhận dạng dựa trên mô hình đã train
def predict(imgPath, model):
# chạy hàm tiền xử lý ảnh
    img = preImg(imgPath)
# nhận dạng ảnh dựa trên xem xét các đặc trưng của ảnh so với các đặc trưng có trong mô hình
    predicted = model.predict(img)
# lấy kết quả là label có tỉ lệ dự đoán cao nhất
    predictedLabel = numpy.argmax(predicted)
    return predictedLabel


#################################################################################################################################


# nhận dạng
def classify(imgPath):
    print("dạnh sách dataset:")
    dirList=[i for i in os.listdir(".") if i.startswith("ds")]
    for i in dirList:
        print(dirList.index(i)+1,": "+i)
    dataIndex=inumpyut("nhập số thứ tự của dataset: ")
    dataFolder="./"+dirList[int(dataIndex)-1]
    allLabels=os.listdir(dataFolder)

    logr=open("train.log", "r")
    lastTrained=logr.readlines()
    while True:
        if lastTrained == []:
            loop=int(inumpyut("nhập số lần train(để tối ưu khả năng nhận dạng): "))
            trainClassify(dataFolder, allLabels, loop)
            logw=open("train.log", "a")
            logw.write(dataIndex+"\n")
            logw.close()
            break
        elif lastTrained[-1].strip()==dataIndex:
            print("mô hình nhận dạng", dirList[int(dataIndex)-1], "đã tồn tại, bạn có muốn train lại? Y/n:")
            trainAsk=inumpyut()
            if trainAsk=="y" or trainAsk =="Y":
    # train mô hình
                loop=int(inumpyut("nhập số lần train(để tối ưu khả năng nhận dạng): "))
                trainClassify(dataFolder, allLabels, loop)
                logw=open("train.log", "a")
                logw.write(dataIndex+"\n")
                logw.close()
                break
            elif trainAsk=="n" or trainAsk =="N":
                break
            else:
                print("bạn đã nhập sai, mời nhập lại")
                continue
        else:
            loop=int(inumpyut("nhập số lần train(để tối ưu khả năng nhận dạng): "))
            trainClassify(dataFolder, allLabels, loop)
            logw=open("train.log", "a")
            logw.write(dataIndex+"\n")
            logw.close()
            break
    logr.close()
    
# Load mô hình đã train
    csfModel = load_model('classify.keras')
    
# gọi hàm nhận dạng
    predictedResult = predict(imgPath, csfModel)

# in kết quả
    print("con này là con:",allLabels[predictedResult])

# print("danh sách test img: ")
# for i in os.listdir("./test"):
#     print(os.listdir("./test").index(i)+1,": "+i)
# chooseImg=inumpyut("chọn ảnh: ")
# imgPath="./test/"+os.listdir("./test")[int(chooseImg)-1]
# classify(imgPath)

# Nhập thông tin từ người dùng =>  link tập dataset
imgTest="./test/dog.webp"
datasetFolder="./PetImages/Cat"
# Tạo hình ảnh mới dựa trên inumpyut_data và dataset_images
create_img(imgTest, datasetFolder)
