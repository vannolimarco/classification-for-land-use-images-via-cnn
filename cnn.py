import os
import  argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import  pathconfig
from preprocessing import  read_img_from_location,data_normaization
from model import save_model,load_model, confusion_matrix,plots_loss_accuracy_from_training

paths = pathconfig.paths()
def main(mode,dataset_train,dataset_val,dataset_test):
    path_save_model = paths.MODEL_CNN
    path_save_weights = paths.WEIGHTS_CNN

    img_rows = 256
    img_cols = 256
    num_class = 21

    # read image from folder FLOW in where there is train,validate and train dataset splitted
    if(mode == 'train'):
        x_train, y_train = read_img_from_location(dataset_train)
        x_val, y_val = read_img_from_location(dataset_val)
        # data normalization
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 3)
        x_train, y_train= data_normaization(x=x_train, y=y_train)
        x_val, y_val = data_normaization(x=x_val, y=y_val)
    elif(mode == 'evaluation'):
        try:
           x_test, y_test = read_img_from_location(dataset_test)
        except:
            print('Check to have the folder called "flow" in order to have dataset for testing/training/validate')
        # data normalization
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        x_test, y_test = data_normaization(x=x_test, y=y_test)

    # the shape of input
    input_shape = (img_rows, img_cols, 3)

    def train_cnn(x_train, y_train, x_val, y_val):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_class, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_fit_history = model.fit(x_train, y_train, batch_size=50, epochs=100, verbose=1,validation_data=(x_val, y_val))
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        plots_loss_accuracy_from_training(model_fit_history)
        save_model(model, path_model=path_save_model, path_weights=path_save_weights)

    if(mode == 'train'):
        class_names = os.listdir(dataset_train)  # the class names from dataset
        train_cnn(x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val)
        loaded_model = load_model(path_save_model, path_save_weights)
        confusion_matrix(loaded_model, x_val, y_val, class_names)
    else:
        #load the model from cnn
        try:
            class_names = os.listdir(dataset_test)  # the class names from dataset
        except:
            print('Check to have the folder called "flow" in order to have dataset for testing/training/validate')

        loaded_model = load_model(path_save_model, path_save_weights)
        # evaluation of results from VGG model (confusion matrix)
        confusion_matrix(loaded_model, x_test, y_test, class_names)



if __name__ == "__main__":
    paths = pathconfig.paths()
    parser = argparse.ArgumentParser(description='Python with Multi-modal')
    parser.add_argument('--mode', default='evaluation', type=str, help='support option: train/evaluation')
    parser.add_argument('--path_dataset_test', default=paths.FLOW_TEST, type=str,help='path of dataset: path')
    parser.add_argument('--path_dataset_train', default=paths.FLOW_TRAIN, type=str,help='path of dataset: path')
    parser.add_argument('--path_dataset_val', default=paths.FLOW_VALIDATE, type=str,help='path of dataset: path')
    args = parser.parse_args()

    if ((args.mode == 'evaluation' or args.mode == 'train')):  #python cnn.py --mode evaluation --path_dataset_test "the path of dataset fro testing" (cmd)
        main(mode=args.mode, dataset_test=args.path_dataset_test,dataset_train=args.path_dataset_train,dataset_val=args.path_dataset_val)
    else:
        print('ERROR: check mode arg {}. mode args can be: train or evaluation'.format(args.mode))