
# coding: utf-8

# In[29]:


from keras.utils import np_utils
import numpy as np


# In[30]:


from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()


# In[31]:


x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')


# In[32]:


x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255


# In[33]:


y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)


# In[34]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[90]:


model = Sequential()


# In[91]:


model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))


# In[92]:


model.add(Dropout(0.75))


# In[93]:


model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))


# In[94]:


print(model.summary())


# In[95]:


model.compile(loss='categorical_crossentropy',
              optimizer = 'adam' , metrics=['accuracy'])


# In[96]:


train_history=model.fit(x=x_Train_normalize,
                        y=y_Train_OneHot,validation_split=0.2,
                        epochs=15, batch_size=200,verbose=2)


# In[97]:


import matplotlib.pyplot as plt
def show_train_history(train_history,train,vlidation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[vlidation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train' , 'vlidation'], loc='upper left')
    plt.show()


# In[98]:


show_train_history(train_history,'acc','val_acc')


# In[99]:


show_train_history(train_history,'loss','val_loss')


# In[100]:


scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy=',scores[1])


# In[101]:


prediction = model.predict_classes(x_Test)


# In[102]:


prediction


# In[103]:


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


# In[104]:


plot_images_labels_prediction(x_test_image,y_test_label,
                               prediction,idx=340,num=1)


# In[105]:


import pandas as pd
pd.crosstab(y_test_label,prediction,
            rownames=['label'],colnames=['predict'])

