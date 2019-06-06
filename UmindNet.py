#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:05:33 2018

@author: sasmacjc
"""

import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data_gf import *
import metrics

class myMultiUNet(object):
    
    def __init__(self,img_rows = 256,img_cols = 256):
        
        self.img_rows = img_rows
        self.img_cols = img_cols
        
    def load_data(self):
        
        mydata = dataProcess(self.img_rows,self.img_cols)
        
        imgs_train,imgs_mask_train = mydata.load_train_data()
        return imgs_train, imgs_mask_train
    
    def MultiUnet(self):
        # Deep Feature Extraction
        inputs = Input((self.img_rows,self.img_cols,3))
        conv1_1 = Conv2D(64,3,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(inputs)
        conv1_1 = Conv2D(64,3,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv1_1)
        pool1_1 = MaxPooling2D(pool_size=(2,2))(conv1_1)

        conv1_2 = Conv2D(128,3,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool1_1)
        conv1_2 = Conv2D(128,3,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv1_2)
        pool1_2 = MaxPooling2D(pool_size=(2,2))(conv1_2)
        
        conv1_3 = Conv2D(256,3,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool1_2)
        conv1_3 = Conv2D(256,3,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv1_3)
        pool1_3 = MaxPooling2D(pool_size=(2,2))(conv1_3)
        print( "conv1_3 shape:"),conv1_3.shape
 
        conv2_1 = Conv2D(64,5,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(inputs)
        conv2_1 = Conv2D(64,5,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv2_1)
        pool2_1 = MaxPooling2D(pool_size=(2,2))(conv2_1)       

        conv2_2 = Conv2D(128,5,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool2_1)
        conv2_2 = Conv2D(128,5,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv2_2)
        pool2_2 = MaxPooling2D(pool_size=(2,2))(conv2_2)
        
        conv2_3 = Conv2D(256,5,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool2_2)
        conv2_3 = Conv2D(256,5,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv2_3)
        pool2_3 = MaxPooling2D(pool_size=(2,2))(conv2_3)

        print( "conv2_3 shape:"),conv2_3.shape

        conv3_1 = Conv2D(64,7,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(inputs)
        conv3_1 = Conv2D(64,7,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv3_1)
        pool3_1 = MaxPooling2D(pool_size=(2,2))(conv3_1)            

        conv3_2 = Conv2D(128,7,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool3_1)
        conv3_2 = Conv2D(128,7,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv3_2)
        pool3_2 = MaxPooling2D(pool_size=(2,2))(conv3_2)
        
        conv3_3 = Conv2D(256,7,activation ='relu',padding='same',kernel_initializer = 'he_normal' )(pool3_2)
        conv3_3 = Conv2D(256,7,activation = 'relu',padding='same',kernel_initializer='he_normal')(conv3_3)
        pool3_3 = MaxPooling2D(pool_size=(2,2))(conv3_3)

        print( "conv3_3 shape:"),conv3_3.shape        
       
        merge_left = concatenate([pool1_3,pool2_3],axis = 3)
        merge_left = concatenate([merge_left,pool3_3],axis =3)
        #multiscale perception
        conv4_1 = Conv2D(512,3,activation = 'relu',padding='same',dilation_rate = (6,6),kernel_initializer='he_normal')(merge_left)
        conv4_1 = Conv2D(512,1,activation = 'relu',padding='same')(conv4_1)
        drop4_1 = Dropout(0.5)(conv4_1)
        
        conv4_2 = Conv2D(512,3,activation = 'relu',padding='same',dilation_rate = (12,12),kernel_initializer='he_normal')(drop4_1)
        conv4_2 = Conv2D(512,1,activation = 'relu',padding='same')(conv4_2)
        drop4_2 = Dropout(0.5)(conv4_2)

        
        conv4_3 = Conv2D(512,3,activation = 'relu',padding='same',dilation_rate = (15,15),kernel_initializer='he_normal')(drop4_2)
        conv4_3 = Conv2D(512,1,activation = 'relu',padding='same')(conv4_3)
        drop4_3 = Dropout(0.5)(conv4_3)

        
        conv4_4 = Conv2D(512,3,activation = 'relu',padding='same',dilation_rate = (24,24),kernel_initializer='he_normal')(drop4_3)
        conv4_4 = Conv2D(512,1,activation = 'relu',padding='same')(conv4_4)
        merge_mid = Dropout(0.5)(conv4_4)
        
#        merge_mid1 = merge([drop4_1,drop4_2],mode = 'concat',concat_axis = 3)
#        merge_mid2 = merge([drop4_3,drop4_4],mode = 'concat',concat_axis = 3)
#        merge_mid = merge([merge_mid1,merge_mid2],mode = 'concat',concat_axis = 3)
        
        #dense prediction 
        up5_1 = Conv2D(256,3,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(merge_mid))
        merge5_1 = concatenate([up5_1,conv1_3],axis=3)
        conv5_1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5_1)
        conv5_1 = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv5_1)
#        conv5_1 = Conv2D(1,1,activation='relu',padding='same')(conv5_1)
        
        up6_1 = Conv2D(128,3,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5_1))
        merge6_1 = concatenate([up6_1,conv1_2],axis=3)
        conv6_1 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6_1)
        conv6_1 = Conv2D(128,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv6_1)
#        conv6_1 = Conv2D(1,1,activation='relu',padding='same')(conv6_1)       
        
        up7_1 = Conv2D(64,3,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6_1))
        merge7_1 = concatenate([up7_1,conv1_1],axis=3)
        conv7_1 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7_1)
        conv7_1 = Conv2D(64,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_1)
        conv7_1 = Conv2D(1,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_1)
     
        up5_2 = Conv2D(256,5,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(merge_mid))
        merge5_2 = concatenate([up5_2,conv2_3],axis=3)
        conv5_2 = Conv2D(256,5,activation='relu',padding='same',kernel_initializer='he_normal')(merge5_2)
        conv5_2 = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv5_2)
#        conv5_1 = Conv2D(1,1,activation='relu',padding='same')(conv5_1)
        
        up6_2 = Conv2D(128,5,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5_2))
        merge6_2 = concatenate([up6_2,conv2_2],axis=3)
        conv6_2 = Conv2D(128,5,activation='relu',padding='same',kernel_initializer='he_normal')(merge6_2)
        conv6_2 = Conv2D(128,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv6_2)
#        conv6_1 = Conv2D(1,1,activation='relu',padding='same')(conv6_1)       
        
        up7_2 = Conv2D(64,5,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6_2))
        merge7_2 = concatenate([up7_2,conv2_1],axis=3)
        conv7_2 = Conv2D(64,5,activation='relu',padding='same',kernel_initializer='he_normal')(merge7_2)
        conv7_2 = Conv2D(64,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_2)
        conv7_2 = Conv2D(1,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_2)
        
        up5_3 = Conv2D(256,7,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(merge_mid))
        merge5_3 = concatenate([up5_3,conv3_3],axis=3)
        conv5_3 = Conv2D(256,7,activation='relu',padding='same',kernel_initializer='he_normal')(merge5_3)
        conv5_3 = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv5_3)
#        conv5_1 = Conv2D(1,1,activation='relu',padding='same')(conv5_1)
        
        up6_3 = Conv2D(128,7,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv5_3))
        merge6_3 = concatenate([up6_3,conv3_2],axis=3)
        conv6_3 = Conv2D(128,7,activation='relu',padding='same',kernel_initializer='he_normal')(merge6_3)
        conv6_3 = Conv2D(128,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv6_3)
#        conv6_1 = Conv2D(1,1,activation='relu',padding='same')(conv6_1)       
        
        up7_3 = Conv2D(64,7,activation='relu',padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6_3))
        merge7_3 = concatenate([up7_3,conv3_1],axis=3)
        conv7_3 = Conv2D(64,7,activation='relu',padding='same',kernel_initializer='he_normal')(merge7_3)
        conv7_3 = Conv2D(64,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_3)
        conv7_3 = Conv2D(1,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv7_3) 
        
        merge_right = concatenate([conv7_1,conv7_2],axis =3)
        merge_right = concatenate([merge_right,conv7_3],axis=3)
        conv8 = Conv2D(1,1,activation='sigmoid')(merge_right)
        
        model = Model(input = inputs,output = conv8)
        model.compile(optimizer = Adam(lr=1e-4),loss="binary_crossentropy",metrics=['accuracy',metrics.precision,metrics.recall,metrics.f1])
        
        return model
    
    def train(self):
        imgs_train,imgs_mask_train = self.load_data()
        
        model = self.MultiUnet()
        model_checkpoint = ModelCheckpoint('multiUnetgf2.hdf5',monitor='loss',verbose=1,save_best_only=True)
        
        hist = model.fit(imgs_train,imgs_mask_train,batch_size=2,nb_epoch=100,verbose=1,validation_split=0.2,shuffle=True,callbacks=[model_checkpoint])
        
        with open('multi-netgf2.txt','w') as f:
            f.write(str(hist.history))
            

if __name__=='__main__':
    net = myMultiUNet()
    net.train()
        
        
        
        

        