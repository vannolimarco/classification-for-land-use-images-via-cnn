from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import  matplotlib
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

def load_model(path_model, path_weight):
    # load json and create model
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_weight)
    return loaded_model
    print("Loaded model from disk")

def save_model(model, path_model, path_weights):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path_model, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path_weights)
    print("Saved model to disk")


# Predict on the test images
def confusion_matrix(model, x_test_data, y_test_data,class_names):
    y_pred = model.predict_classes(x_test_data, verbose=0)

    # Flatten Y into a vector
    y_test = np.nonzero(y_test_data)[1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model predication accuracy: {accuracy:.3f}')
    print(f'\nClassification report:\n {classification_report(y_test, y_pred)}')

    y_pred = model.predict_classes(x_test_data)
    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,index=class_names,columns=class_names)

    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plots_loss_accuracy_from_training(model_fit_history):
        matplotlib.style.use('seaborn')
        epochs = len(model_fit_history.history['loss'])
        max_loss = max(max(model_fit_history.history['loss']), max(model_fit_history.history['val_loss']))
        plt.axis([0, epochs+1, 0, round(max_loss * 2.0) / 2 + 0.5])
        x = np.arange(1, epochs+1)
        plt.plot(x, model_fit_history.history['loss'])
        plt.plot(x, model_fit_history.history['val_loss'])
        plt.title('Training loss vs. Validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='right')
        plt.show()

        matplotlib.style.use('seaborn')
        epochs = len(model_fit_history.history['accuracy'])
        plt.axis([0, epochs+1, 0, 1.2])
        x = np.arange(1, epochs+1)
        plt.plot(x, model_fit_history.history['accuracy'])
        plt.plot(x, model_fit_history.history['val_accuracy'])
        plt.title('Training accuracy vs. Validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='right')
        plt.show()

