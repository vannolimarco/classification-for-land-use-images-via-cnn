import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import  argparse
import tensorflow as tf
import pandas as pd
import  pathconfig
import seaborn as sns
from skimage.io import imread, imsave
from preprocessing import create_train_test_validate_dataset,get_bottleneck_features
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from model import save_model,load_model,confusion_matrix,plots_loss_accuracy_from_training


def main(mode,dataset_eva):
        def build_fully_connected(input_shape, num_classes):
            """
            Create a fully-connected model to train or test on UC Merced dataset.
            """
            model = Sequential()
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
            return model


        def train_model_VGG(x_train,y_train,x_validate,y_validate,num_classes):
                # Build, compile, and fit the model
                model = build_fully_connected(input_shape=X['train'].shape[1:], num_classes=num_classes)
                adam = optimizers.Adam(lr=0.0001)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                model_fit_history = model.fit(X['train'], Y['train'], batch_size=64, epochs=50,
                                              verbose=2, validation_data=(X['validate'], Y['validate']))

                epochs = np.argmin(model_fit_history.history['val_loss']) + 1
                print(f'Stop training at {epochs} epochs')

                plots_loss_accuracy_from_training(model_fit_history) # plots for loss and accuracy model after training

                # Merge training and validation data
                X_train = np.concatenate([x_train,x_validate])  #concatenate train dataset
                Y_train = np.concatenate([y_train, y_validate]) #concatenate validation dataset

                # Randomly shuffle X and Y
                shuffle_index = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_index]
                Y_train = Y_train[shuffle_index]
                model = build_fully_connected(input_shape=X_train.shape[1:], num_classes=num_classes)
                model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print('Train with Training dataset + Validation dataset as input.')
                model_fit_history = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, verbose=0)  # train with trainign and validation dataset
                save_model(model,path_save_model_vgg,path_save_weight_vgg)



        paths = pathconfig.paths() #object from class pathconfig to extract the path for solution

        path_save_model_vgg = paths.MODEL_VGG  #the path of model vgg
        path_save_weight_vgg = paths.WEIGHTS_VGG #the path of weights vgg

        # Collect class names from directory names in './data/UCMerced_LandUse/Images/'
        sources_dataset = paths.FLOW_TRAIN  #the sources path of dataset
        class_names = os.listdir(sources_dataset) # the class names from dataset
        try:
           target_dirs = {target: os.path.join(paths.BASE_FLOW, target) for target in ['train', 'validate', 'test']}
        except:
            print('Check to have the folder called "flow" in order to have dataset for testing/training/validate')

        #Calculate the training image means by channel (3)
        means = []
        for root, _, filenames in os.walk(target_dirs['train']):
            for filename in filenames: # from filenames of trainng directory of flow
                filepath = os.path.join(root, filename)
                image = imread(filepath) #read image
                means.append(np.mean(image, axis=(0, 1)))
        channel_means = np.mean(means, axis=0)

        # Let's try the VGG16 model from keras
        pretrained_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

        # Extract bottleneck features from pretrained model, predicting from "dataset" directory
        # the num of classes (21)
        num_classes = len(class_names)  #num of classes
        X, Y = dict(), dict()   #the two dict for inputs and outputs
        preprocess = lambda x: x - channel_means
        for dataset in ['train', 'validate', 'test']:
            X[dataset], Y[dataset] = get_bottleneck_features(model= pretrained_model, dataset=dataset,target_dirs=target_dirs,preproc_func= preprocess)


        # for re-training the VGG , decomment the code line "train_model_VGG"
        if(mode == 'train'):
           train_model_VGG(X['train'],Y['train'],X['validate'],Y['validate'],num_classes)

        # load the model from VGG
        loaded_model = load_model(path_save_model_vgg,path_save_weight_vgg)

        #evaluation of results from VGG model (confusion matrix)
        if(dataset_eva == 'test'):
            confusion_matrix(loaded_model,X['test'],Y['test'],class_names)
        elif(dataset_eva == 'train'):
            confusion_matrix(loaded_model, X['train'], Y['train'], class_names)
        else:
            confusion_matrix(loaded_model, X['validate'], Y['validate'], class_names)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python with Multi-modal')
    parser.add_argument('--mode', default='evaluation', type=str, help='support option: train/evaluation')
    parser.add_argument('--dataset', default='test', type=str,help='dataset: train/test/val')
    args = parser.parse_args()

    if ((args.mode == 'evaluation' or args.mode == 'train') and (args.dataset == 'train' or args.dataset == 'test' or args.dataset == 'val')):  #python vgg.py --mode evaluation --dataset test (cmd)
        main(mode=args.mode, dataset_eva=args.dataset)
    else:
        if((args.mode != 'evaluation' or args.mode != 'train') and (args.dataset == 'train' or args.dataset == 'test' or args.dataset == 'val') ):
            print('ERROR: check mode arg {}. mode args can be: train or evaluation'.format(args.mode))
        elif((args.dataset != 'train' or args.dataset != 'test' or args.dataset != 'val') and (args.mode == 'evaluation' or args.mode == 'train')):
            print('ERROR: check dataset arg {}. dataset args can be: train,test or val'.format(args.dataset))
        else:
            print('ERROR: check mode arg {}. mode args can be: train or evaluation'.format(args.mode))
            print('ERROR: check dataset arg {}. dataset args can be: train,test or val'.format(args.dataset))

