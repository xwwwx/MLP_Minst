
# coding: utf-8




from keras.utils import np_utils
import numpy as np


# 下載Minst資料


from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()


# 將資料轉換成1維float32陣列


x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')


# 資料標準化


x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255


# 一位有效編碼


y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)





from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# 初始化模型


model = Sequential()


# 加入輸入層 units=神經元數量 input_dim=輸入資料長度 activation=激活函數


model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))


# 加入Dropout層 


model.add(Dropout(0.75))


# 加入輸出層


model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))


# 顯示模型概要


print(model.summary())


# 訓練初始化 loss=損失函數 optimizer=最佳化方法


model.compile(loss='categorical_crossentropy',
              optimizer = 'adam' , metrics=['accuracy'])


# 訓練模型 validation_split=分隔訓練資料及驗證資料的百分比 epochs=訓練次數 batch_size=每次訓練幾筆資料 


train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot,validation_split=0.2,
                        epochs=15, batch_size=200,verbose=2)


# 畫圖函數


import matplotlib.pyplot as plt
def show_train_history(train_history,train,vlidation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[vlidation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train' , 'vlidation'], loc='upper left')
    plt.show()


# 畫出訓練準確度及驗證準確度的曲線


show_train_history(train_history,'acc','val_acc')


# 畫出訓練損失函數及驗證損失函數的曲線


show_train_history(train_history,'loss','val_loss')


# 評估準確率


scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])


# 預測


prediction = model.predict_classes(x_Test)


# 畫圖函數


def plot_images_labels_prediction(images,labels,
                                 prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25 : num=25
    for i in range(num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap='binary')
        title='label='+str(labels[idx])
        if len(prediction) > 0:
            title+=',predict='+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()


# 畫出預測結果


plot_images_labels_prediction(x_test_image,y_test_label,
                               prediction,idx=340,num=1)


# 混淆陣列


import pandas as pd
pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])

