from preprocessing import pipeline
from train import train_base, train_adversarial # , train_local
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
# import argparse
import copy


def prepare_data_for_training(data_df, batch_size, split, save_path):
    numpy_df = data_df.to_numpy()
    new_numpy_df = []
    for i in range(numpy_df.shape[0]):
        # Flatten the numpy list into a single list
        flattened_list = [item for sublist in numpy_df[i] for item in (sublist if isinstance(sublist, list) else [sublist])]
        new_numpy_df.append(flattened_list)
    new_numpy_df = np.array(new_numpy_df)

    tf_df = pd.DataFrame(new_numpy_df)

    # Save the dataframe
    tf_df.to_csv(f'{save_path}/data-tf.csv', index=True)

    features = tf_df.loc[:, :(len(tf_df.columns)-2)]
    labels = tf_df.iloc[:, -1:]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Split the dataset into training and testing sets (e.g., 80% train, 20% test)
    train_size = int(split * len(dataset))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    # Shuffle and batch
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def get_model(input_size, model):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    if model == 'small':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dense(2, activation='linear', name='output_layer')
        ])

    if model == 'mid':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])

    if model == 'mid2':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.Dense(2, activation='linear', name='output_layer')
        ])

    if model == 'mid3':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.Dense(256, activation='relu', name='dense_3'),
            layers.Dense(128, activation='relu', name='dense_4'),
            layers.Dense(2, activation='linear', name='output_layer')
        ])
    
    if model == 'mid4':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(512, activation='relu', name='dense_0'),
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.Dense(2, activation='linear', name='output_layer')
        ])

    if model == 'big':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_3'),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])

    if model == 'big2':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])

    if model == 'big3':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])
    
    if model == 'big4':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            layers.Dense(2048, activation='relu', kernel_initializer=initializer, name='dense_1'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])

    if model == 'massive':
        model = keras.Sequential([
            layers.Input(shape=(input_size,), name='input_features'),
            #layers.Dense(8182, activation='relu', kernel_initializer=initializer, name='dense_1'),
            #layers.Dense(4096, activation='relu', kernel_initializer=initializer, name='dense_2'),
            layers.Dense(2048, activation='relu', kernel_initializer=initializer, name='dense_3'),
            layers.Dense(1024, activation='relu', kernel_initializer=initializer, name='dense_4'),
            layers.Dense(512, activation='relu', kernel_initializer=initializer, name='dense_5'),
            layers.Dense(256, activation='relu', kernel_initializer=initializer, name='dense_6'),
            layers.Dense(128, activation='relu', kernel_initializer=initializer, name='dense_7'),
            layers.Dense(2, activation='linear', kernel_initializer=initializer, name='output_layer')
        ])

    print(model.summary())
    return model

def base_train():
    # Datasets
    pos_train = "CIC2017"
    neg_train = "CIC2017" # // CIC 2017 but just HULK -> Generalise to different networks + page sizes (CIC 2018 Hulk + DetGen Hulk)
                          # // CIC 2017 but just HULK -> Generalise different attack categories (Rest of CIC 2017 Dos + CIC 2018 Dos)

    pos_test = "CIC2018"
    neg_test = "DetGenSSH"


    preprocess_dict={"time_max": 50000000000,
                     "iat_max": 5000000000,
                     "size_max": 1000,
                     "flag_max": 256}

    batch_size = 64
    pkts_length = 10
    input_size = 2 + pkts_length*4
    epochs = 150
    resample=None
    save=True
    load=False
    model_save_path = './models'
    attack_name = 'DoS'

    train_dataset, test_dataset = pipeline(pos_train,
                                           neg_train,
                                           pos_test,
                                           neg_test,
                                           resample=resample,
                                           m=pkts_length,
                                           batch_size=batch_size,
                                           preprocess_dict=preprocess_dict,
                                           save=save,
                                           load=load)

    model_base = get_model(input_size=input_size, model='small')
    model_base = train_base(model_base, epochs, train_dataset, test_dataset, model_save_path, attack_name)

if __name__ == '__main__':
    # Datasets
    pos_train = "CIC2017"
    neg_train = "CIC2017" # // CIC 2017 but just HULK -> Generalise to different networks + page sizes (CIC 2018 Hulk + DetGen Hulk)
                          # // CIC 2017 but just HULK -> Generalise different attack categories (Rest of CIC 2017 Dos + CIC 2018 Dos)

    pos_test = "CIC2018"
    neg_test = "DetGenSSH"


    preprocess_dict={"time_max": 50000000000,
                     "iat_max": 5000000000,
                     "size_max": 1000,
                     "flag_max": 256}

    pkts_length = 10
    input_size = 2 + pkts_length*4
    epochs = 64
    resample=None
    save=False
    load=True
    model_save_path = './models'
    attack_name = 'DoS2'
    from_logits = True
    pgd_steps = 5

    

    # With time elapsed = 0.0
    goodHTTP1 =         [[0.0, 0.0], [0.0, 0.0], # Time elapsed, protocol
                        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [2/256, 2/256], [18/256, 18/256], [16/256, 16/256], [24/256, 24/256], [16/256, 16/256], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], # Pkd IATs
                        [52/1000, 52/1000], [52/1000, 52/1000], [40/1000, 40/1000], [100/1000, 500/1000], [40/1000, 40/1000], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size

    # With time elapsed between [0.001, 1.0]
    goodHTTP2 =         [[0.002, 1.0], [0.0, 0.0], # Time elapsed, protocol
                        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [2/256, 2/256], [18/256, 18/256], [16/256, 16/256], [24/256, 24/256], [16/256, 16/256], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], [0.000001, 0.05], # Pkd IATs
                        [52/1000, 52/1000], [52/1000, 52/1000], [40/1000, 40/1000], [100/1000, 500/1000], [40/1000, 40/1000], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size

    invalid =          [[0.0, 1.0], [0.0, 1.0], # Time elapsed, protocol
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkd IATs
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size
    
    hulk =             [[0.00000000000001, 0.001], [0.0, 0.0], # Time elapsed, protocol
                        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [2/256, 2/256], [18/256, 18/256], [16/256, 16/256], [24/256, 24/256], [16/256, 16/256], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkd IATs
                        [52/1000, 52/1000], [52/1000, 52/1000], [40/1000, 40/1000], [100/1000, 500/1000], [40/1000, 40/1000], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size



    SYNFlood =         [[0.0, 1.0], [0.0, 1.0], # Time elapsed, protocol
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkd IATs
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size
    
    slowhttptest =     [[0.0, 1.0], [0.0, 0.0], # Time elapsed, protocol
                        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [2/256, 2/256], [18/256, 18/256], [16/256, 16/256], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkd IATs
                        [52/1000, 52/1000], [52/1000, 52/1000], [40/1000, 40/1000], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size
    
    slowIATsAttacks =  [[0.00000000000001, 0.001], [0.0, 0.0], # Time elapsed, protocol
                        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt direction
                        [2/256, 2/256], [18/256, 18/256], [16/256, 16/256], [24/256, 24/256], [16/256, 16/256], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], # Pkt flags
                        [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], [0, 1.0], # Pkd IATs
                        [52/1000, 52/1000], [52/1000, 52/1000], [40/1000, 40/1000], [100/1000, 500/1000], [40/1000, 40/1000], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # Pkt size

    model_name = 'mid'
    model_names = ['big4']
    batch_sizes = [64]
    #alfas = [0.0001]
    #alfas = [0.00001]
    alfas=[0.05, 0.02, 0.01]
    
    for batch_size in batch_sizes:

        hyperrectangles = []
        hyperrectangles_labels = []

        for _ in range(int((batch_size/36) + 1)):
            for _ in range(9):
                hyperrectangles.append(goodHTTP1)
                hyperrectangles.append(goodHTTP2)
                hyperrectangles_labels.append(0)
                hyperrectangles_labels.append(0)

            for _ in range(9):
                hyperrectangles.append(hulk)
                hyperrectangles_labels.append(1)

            for i in range(9):
                temp = copy.deepcopy(slowIATsAttacks)
                temp[i+23][0] = 0.06
                hyperrectangles.append(temp)
                hyperrectangles_labels.append(1)

        hyperrectangles = np.array(hyperrectangles)
        hyperrectangles_labels = np.array(hyperrectangles_labels)

        train_dataset, valid_dataset, test_dataset = pipeline(pos_train,
                                            neg_train,
                                            pos_test,
                                            neg_test,
                                            resample=resample,
                                            m=pkts_length,
                                            batch_size=batch_size,
                                            preprocess_dict=preprocess_dict,
                                            save=save,
                                            load=load,
                                            attack_name=attack_name)
        for model_ in model_names:
            for alfa in alfas:
                # Adversarial model
                print(f"=========== Alfa Value: {alfa} ===========")
                model_adv = get_model(input_size=input_size, model=model_)
                #model_adv = train_local(model_adv, 40, train_dataset, test_dataset, "./sqli_models/", "SQLi_trunc")
                model_adv = train_adversarial(model_adv, train_dataset, valid_dataset, test_dataset, epochs, batch_size, pgd_steps, hyperrectangles, hyperrectangles_labels, alfa, from_logits, model_save_path, attack_name, model_)

