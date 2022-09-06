import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ks
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Mapping from nucleotides to integers
NUCLEOTIDE_MAPPING = {'A': 0, 'T' : 1, 'C' : 2, 'G' : 3}

#Set configuration for GPU
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

"""
Parameters:
    -results: Dictionary containing evaluation results

Build results table, which is shown and saved as a .csv file.
"""
def show_results(results):
    result_table = pd.DataFrame.from_dict(results)
    print(result_table)
    result_table.to_csv('evaluation_results.csv', index=True)


"""
Parameters:
    -history: loss- and metric data over training epochs
    -model_name: name of the model

Plots the training- and validation accuracy over training epochs.
"""
def plot_history(history, model_name='model'):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Training epochs')
    plt.ylabel('Accuracy score')
    plt.title('%s model accuracy over training' % (model_name))
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(model_name+'_accuracy_plot', dpi=300)
    plt.close()


"""
Parameters:
    -type: Model type to build (embedded/normal)

Constructs CNN model to predict TF binding based on DNA sequence data.

"""
def build_model(type=''):
    if type=='embedded':
        model = Sequential(
        [
        #Embeds DNA sequences into a 1 X 1500 X 4 vector
        layers.Embedding(input_dim=4, output_dim=16, input_length=1500),
        #Apply 1D-convolution using a 8-sized kernel
        layers.Conv1D(80, 8, activation='relu'),
        #Followed by average pooling with a 4-sized kernel
        layers.AveragePooling1D(4),
        #And dropout for regularization
        layers.Dropout(0.2),
        #Repeat steps above
        layers.Conv1D(160, 8, activation='relu'),
        layers.AveragePooling1D(4),
        layers.Dropout(0.2),
        layers.Conv1D(320, 8, activation='relu'),
        layers.Dropout(0.4),
        #Flatten the data
        layers.Flatten(),
        #End with fully connected layer and sigmoid output.
        layers.Dense(720, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ]
    )
    elif type=='normal':
        model = Sequential(
        [
        ks.Input(shape=(1500,4)),
        #Apply 1D-convolution using a 8-sized kernel
        layers.Conv1D(80, 8, activation='relu'),
        #Followed by average pooling with a 4-sized kernel
        layers.AveragePooling1D(4),
        #And dropout for regularization
        layers.Dropout(0.2),
        #Repeat steps above
        layers.Conv1D(160, 8, activation='relu'),
        layers.AveragePooling1D(4),
        layers.Dropout(0.2),
        layers.Conv1D(320, 8, activation='relu'),
        layers.Dropout(0.4),
        #Flatten the data
        layers.Flatten(),
        #End with fully connected layer and sigmoid output.
        layers.Dense(720, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ]
    )
    #Use RMSprop optimization algorithm, use binary crossentropy loss and measure accuracy.
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


"""
Parameters:
    -model: model to be trained
    -train_seq: training dataset consisting of sequences
    -train_y: labels of training dataset
    -valid_seq: validation dataset consisting of sequences
    -valid_y: labels of validation dataset

Trains the model on the training data utilizing the validation data.

Returns:
    -model: Trained model
    -history: Model history (losses, accuracy scores over training epochs)

"""
def train_cnn(model, train_seq, train_y, valid_seq, valid_y):
    #Makes sure the best epoch regarding the validation data is saved and apply early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                                   restore_best_weights=True, min_delta=0.01)
    #Train the model
    history = model.fit(train_seq,
               train_y,
               epochs=100,
               batch_size=128,
               verbose=1,
               validation_data = (valid_seq,
                                   valid_y),
               callbacks=[es_callback]
               )
    return model, history


"""
Parameters:
    -seq: Nucleotide sequence (String)

Converts nucleotide sequence to one-hot-encoding array for use
in the deep learning model.

Returns:
    -one-hot encoded sequence (np matrix)
"""
def one_hot_encoding(seq):
    encoded = np.zeros(1500, dtype=int)
    for pos in range(len(seq)):
        encoded[pos] = NUCLEOTIDE_MAPPING[seq[pos]]
    return ks.utils.to_categorical(encoded)


"""
Parameter:
    -seq: Nucleotide sequence (String)

Converts nucleotide sequence into integer sequence which can
be used for embedding in the embedded tf prediction model.

Returns:
    -integer sequence representing the nucleotide sequence (np array)
"""
def seq_to_int(seq):
    encoded = np.zeros(1500, dtype=int)
    for pos in range(len(seq)):
        encoded[pos] = NUCLEOTIDE_MAPPING[seq[pos]]
    return encoded


"""
Parameters:
    type: Type of model to be trained (normal/embedded)
    model_name: Name to be given to the model
    results: Dictionary in which the evalution results can be saved
    train_seq: training dataset consisting of sequences
    valid_seq: validation dataset consisting of sequences
    test_seq: test dataset consisting of sequences
    train_y: training labels
    valid_y: validation labels
    test_y: test labels

Builds, trains and evaluates a tf-prediction model and saves the model for future
use.

Returns:
    -model: Trained model
    -results: Dictionary with evaluation results
"""
def get_model(type, model_name, results,
              train_seq, valid_seq, test_seq, train_y, valid_y, test_y):
    #Check if model already exists, if so, load existing model
    if model_name not in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        #Construct model
        model = build_model(type)
        #Train model
        model, model_history = train_cnn(model, train_seq, train_y, valid_seq, valid_y)
        #Evaluate model
        evaluation = model.evaluate(x=test_seq, y=test_y, batch_size=128)
        (results[type]['train'], results[type]['val'],
        results[type]['test']) =  (np.max(model_history.history['accuracy']),
                                   np.max(model_history.history['val_accuracy']), evaluation[1])
        plot_history(model_history, type)
        #Save model
        model.save(model_name)
    else:
        model = ks.models.load_model(model_name)
    return model, results


#Load sequence data
data = np.load('hw1_data.npz')
#print(tf.config.experimental.list_physical_devices('GPU'))

#Create dictionary for evaluation results
results = {"normal" : {'train' : 0, 'val' : 0, 'test' : 0},
            "embedded" : {'train' : 0, 'val' : 0, 'test' : 0}}
#Retrieve train-, validation- and test data and labels.
train_seq_key, valid_seq_key, test_seq_key, train_y_key, valid_y_key, test_y_key = data.files
train_seq, valid_seq, test_seq, train_y, valid_y, test_y = (data[train_seq_key], data[valid_seq_key],
                                                            data[test_seq_key], data[train_y_key],
                                                            data[valid_y_key], data[test_y_key])

#Apply one-hot encoding for use in the CNN-model
oh_train_seq, oh_valid_seq, oh_test_seq = (np.array([one_hot_encoding(x) for x in train_seq]),
                                  np.array([one_hot_encoding(x) for x in valid_seq]),
                                  np.array([one_hot_encoding(x) for x in test_seq]))
#train 'normal' CNN model
normal_model, results = get_model('normal', 'normal_tf_prediction_model', results,
                                  oh_train_seq, oh_valid_seq, oh_test_seq, train_y, valid_y, test_y)

#Convert nuc-sequences to integer sequences
int_train_seq, int_valid_seq, int_test_seq = (np.array([seq_to_int(x) for x in train_seq]),
                                              np.array([seq_to_int(x) for x in valid_seq]),
                                              np.array([seq_to_int(x) for x in test_seq]))
#Train embedded CNN model
embedded_model, results = get_model('embedded', 'embedded_tf_prediction_model', results,
                                    int_train_seq, int_valid_seq, int_test_seq, train_y, valid_y, test_y)
#Save and show evaluation results
show_results(results)
