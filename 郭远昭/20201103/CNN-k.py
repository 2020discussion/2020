#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为了学习CNN实例的代码及原理情况
在网上搜索的图像案例进行运行，阅读代码，了解每段代码功能
https://www.cnblogs.com/luoganttcc/p/10525286.html
"""
#模型用到的keras包的调用
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt


#num_classes 代表分类数量
#CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。
#一共包含 10 个类别的 RGB 彩色图 片：飞机（ a叩lane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。
#图片的尺寸为 32×32 ，数据集中一共有 50000 张训练圄片和 10000 张测试图片。

num_classes = 10                                          
model_name = 'cifar10.h5'                                  


#将cifar10数据集载入并划分成训练集、测试集
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#将训练集中的第二张图片显示出来，对于模型没有实质意义，但可以据此直观看到数据集
plt.imshow(x_train[1])
plt.show()

#图片大小为float32相较于float64减少占用时间，相对提高系统运行效率。
#/255是将0~255的像素取值转换至0~1之间，会提升训练效果
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Keras中构建Sequential models顺序模型
model = Sequential()

#网络层
#第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
#对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
#加入relu激活函数
model.add(Conv2D(32, (3, 3), padding='same',strides=(1,1) ,input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
#keras Pool层有个奇怪的地方，stride,默认是(2*2),padding 默认是valid，在写代码是这些参数还是最好都加上
#下面设置了随机失活、最大池化层、激活函数relu和softmax、flatten拉成一列等操作
#Dense即分多少类就有多少Dense节点
model.add(  MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')  )
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()


#原代码中自定义设置了RMSprop优化器，但是运行有问题，我自己改成了直接调用RMSprop或sgd
# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)
# train the model using RMSprop
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#设置训练批次共40回
#保存模型
hist = model.fit(x_train, y_train, epochs=5, shuffle=True)
model.save(model_name)

#评估模型，输出loss和accuracy值
# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print (loss, accuracy)


#5epochs结果
#1.2379807233810425 0.5788999795913696