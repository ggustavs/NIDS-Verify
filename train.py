from tensorflow import keras
# import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import confusion_matrix, f1_score
import neural_structured_learning as nsl

def train_local(model, epochs, train_dataset, test_dataset, save_path, attack_name):


    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.1, adv_step_size=0.005)
    adv_model = nsl.keras.AdversarialRegularization(model, label_keys=["label"], adv_config=adv_config)

        # Compile and train the model
    adv_model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
    )

    
    def convert_to_dictionaries(image, label):
        return {"input": image, "label": label}

    
    train_set_for_adv_model = train_dataset.map(convert_to_dictionaries)
    test_set_for_adv_model = test_dataset.map(convert_to_dictionaries)

    adv_model.fit(train_set_for_adv_model, epochs=epochs, validation_data=test_set_for_adv_model, validation_freq=10)
    
    # Evaluate the model
    results_train = adv_model.evaluate(train_set_for_adv_model)
    results_test = adv_model.evaluate(test_set_for_adv_model)

    print("Train Results: ", results_train)
    print("Test Results: ", results_test)
    
    predictions = np.argmax(adv_model.predict(test_set_for_adv_model), axis=1)
    labels = np.concatenate([y for _, y in test_dataset], axis=0)
    print(confusion_matrix(labels, predictions))
    print("F1: ", f1_score(labels, predictions))

    #print(f"Train acc: {float(train_acc):.4f}, Train loss: {float(train_loss):.4f} --- Test acc: {float(test_acc):.4f}, Test loss: {float(test_loss):.4f}")

    # Save the model
    if not os.path.exists(f'{save_path}/onnx/{attack_name}'):
        os.makedirs(f'{save_path}/onnx/{attack_name}')
    if not os.path.exists(f'{save_path}/tf/{attack_name}'):
        os.makedirs(f'{save_path}/tf/{attack_name}')
    model.save(f'{save_path}/tf/{attack_name}/local_adv_mid')

    return model


def train_base(model, epochs, train_dataset, test_dataset, save_path, attack_name):
    # Compile and train the model
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'
    ],
)
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, validation_freq=1)
    
    # Evaluate the model
    train_loss, train_acc = model.evaluate(train_dataset)
    test_loss, test_acc = model.evaluate(test_dataset)
    
    predictions = np.argmax(model.predict(test_dataset), axis=1)
    labels = np.concatenate([y for _, y in test_dataset], axis=0)
    print(confusion_matrix(labels, predictions))
    print("F1: ", f1_score(labels, predictions))

    print(f"Train acc: {float(train_acc):.4f}, Train loss: {float(train_loss):.4f} --- Test acc: {float(test_acc):.4f}, Test loss: {float(test_loss):.4f}")

    # Save the model
    if not os.path.exists(f'{save_path}/onnx/{attack_name}'):
        os.makedirs(f'{save_path}/onnx/{attack_name}')
    if not os.path.exists(f'{save_path}/tf/{attack_name}'):
        os.makedirs(f'{save_path}/tf/{attack_name}')
    model.save(f'{save_path}/tf/{attack_name}/base_small')

    return model


def train_adversarial(model, train_dataset, valid_dataset, test_dataset, epochs, batch_size, pgd_steps, hyperrectangles, hyperrectangles_labels, alfa, from_logits, save_path, attack_name, model_name):
    if not os.path.exists(f'{save_path}/onnx/{attack_name}'):
        os.makedirs(f'{save_path}/onnx/{attack_name}')
    if not os.path.exists(f'{save_path}/tf/{attack_name}'):
        os.makedirs(f'{save_path}/tf/{attack_name}')
    
    keras.utils.set_random_seed(42)

    # max_adv_acc = 0
    max_epoch = 0

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
    ce_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_batch_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    pgd_attack_single_image_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    adv_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    valid_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    test_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)
    adv_loss_metric = keras.metrics.SparseCategoricalCrossentropy(from_logits=from_logits)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        start_time = time.time()
        # print("[*] Training . . .")
        # Iterate over the batches of the dataset.
        for x_batch_train, y_batch_train in train_dataset:
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Outputs for this minibatch
                ce_loss_value = ce_batch_loss(y_batch_train, outputs)
                ce_loss_value = ce_loss_value * alfa
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(ce_loss_value, model.trainable_variables)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        model.reset_states()
        
        #########################################PGD####################################################
        pgd_dataset = []
        pgd_labels = []
        # print(f"[*] Generating . . . {(time.time() - start_time):.2f}")

        for i, hyperrectangle in enumerate(hyperrectangles):
            t_hyperrectangle = np.transpose(hyperrectangle)

            # Calculate the epsilon for each dimension as ((dim[1] - dim[0]) / (pgd_steps * eps_multiplier))
            eps = []
            for d in hyperrectangle:
                eps.append((d[1] - d[0]) / pgd_steps)
            
            # Generate a pgd point from the hyperrectangle 
            pgd_point = []
            for d in hyperrectangle:
                pgd_point.append(np.random.uniform(d[0], d[1]))
            # PGD attack on the image
            pgd_point = tf.convert_to_tensor([pgd_point], dtype=tf.float32)
            pgd_label = tf.convert_to_tensor([hyperrectangles_labels[i]], dtype=tf.float32)

            for _ in range(pgd_steps):
                with tf.GradientTape() as tape:
                    tape.watch(pgd_point)
                    prediction = model(pgd_point, training=False)
                    pgd_single_image_loss = pgd_attack_single_image_loss(pgd_label, prediction)
                # Get the gradients of the loss w.r.t to the input image.
                gradient = tape.gradient(pgd_single_image_loss, pgd_point)
                # Get the sign of the gradients to create the perturbation
                signed_grad = tf.sign(gradient)
                pgd_point = pgd_point + signed_grad * eps
                # . .  . .                   (pgd_point, pgd_point - delta, pgd_point + delta)
                pgd_point = tf.clip_by_value(pgd_point, t_hyperrectangle[0], t_hyperrectangle[1])
                # print(f"PGD step: {pgd_step + 1}", end="\r")

            # Concatenate the pgd points
            if len(pgd_dataset) > 0:
                pgd_dataset = np.concatenate((pgd_dataset, pgd_point), axis=0)
                pgd_labels = np.concatenate((pgd_labels, pgd_label), axis=0)
            else:
                pgd_dataset = pgd_point
                pgd_labels = pgd_label
        # print(f"[*] Converting . . . {(time.time() - start_time):.2f}")
        pgd_dataset = np.asarray(pgd_dataset)
        pgd_labels = np.asarray(pgd_labels)

        converted_data = pd.DataFrame(pgd_dataset)
        converted_data["Label"] = pgd_labels
        converted_data.to_csv(f"./adv_data_2/{epoch}_adv.csv", index=False)


        # Convert the pgd generated inputs into tf datasets, shuffle and batch them
        pgd_dataset = tf.data.Dataset.from_tensor_slices((pgd_dataset, pgd_labels))
        pgd_dataset = pgd_dataset.shuffle(buffer_size=1024).batch(batch_size)


        # Iterate over the batches of the pgd dataset.
        for x_batch_train, y_batch_train in pgd_dataset: 
            # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                outputs = model(x_batch_train, training=True)  # Outputs for this minibatch
                pgd_loss_value = pgd_batch_loss(y_batch_train, outputs)
                pgd_loss_value = pgd_loss_value * (1 - alfa)
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(pgd_loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        ################################################################################################

       #  print(f"[*] Calculating . . . {(time.time() - start_time):.2f}")
        # Run a training loop at the end of each epoch.
        for x_batch_train, y_batch_train in train_dataset:
            train_outputs = model(x_batch_train, training=False)
            train_acc_metric.update_state(y_batch_train, train_outputs)
            train_loss_metric.update_state(y_batch_train, train_outputs)

        # Run a validation loop at the end of each epoch.
        if valid_dataset is not None:
            for x_batch_test, y_batch_test in valid_dataset:
                test_outputs = model(x_batch_test, training=False)
                valid_acc_metric.update_state(y_batch_test, test_outputs)
                valid_loss_metric.update_state(y_batch_test, test_outputs)
        else:
            valid_acc_metric.update_state([0], [0])
            valid_loss_metric.update_state([0], [0])

        # Run a testing loop at the end of each epoch.
        for x_batch_test, y_batch_test in test_dataset:
            test_outputs = model(x_batch_test, training=False)
            test_acc_metric.update_state(y_batch_test, test_outputs)
            test_loss_metric.update_state(y_batch_test, test_outputs)

        # Run a adv loop at the end of each epoch.
        for x_batch_test, y_batch_test in pgd_dataset:
            adv_outputs = model(x_batch_test, training=False)
            adv_acc_metric.update_state(y_batch_test, adv_outputs)
            adv_loss_metric.update_state(y_batch_test, adv_outputs)

        train_acc = train_acc_metric.result()
        valid_acc = valid_acc_metric.result()
        test_acc = test_acc_metric.result()
        adv_acc = adv_acc_metric.result()
        train_loss = train_loss_metric.result()
        valid_loss = valid_loss_metric.result()
        test_loss = test_loss_metric.result()
        adv_loss = adv_loss_metric.result()

        train_acc_metric.reset_states()
        valid_acc_metric.reset_states()
        test_acc_metric.reset_states()
        adv_acc_metric.reset_states()
        train_loss_metric.reset_states()
        valid_loss_metric.reset_states()
        test_loss_metric.reset_states()
        adv_loss_metric.reset_states()

        train_acc = float(train_acc)
        valid_acc = float(valid_acc)
        test_acc = float(test_acc)
        adv_acc = float(adv_acc)

        train_loss = float(train_loss)
        valid_loss = float(valid_loss)
        test_loss = float(test_loss)
        adv_loss = float(adv_loss)

        print(f"Train acc: {train_acc:.4f}, Train loss: {train_loss:.4f} --- Valid acc: {valid_acc:.4f}, Valid loss: {valid_loss:.4f} --- Test acc: {test_acc:.4f}, Test loss: {test_loss:.4f} --- Adv acc: {adv_acc:.4f}, Adv loss: {adv_loss:.4f} --- Time: {(time.time() - start_time):.2f}s")
        
        if train_acc > 0.7 and (test_acc > 0.7 or valid_acc > 0.7) and adv_acc > 0.5:
            max_epoch = epoch + 1

            model.save(f'{save_path}/tf/{attack_name}/adv_{model_name}_bs{batch_size}_{alfa}_train_{train_acc:.4f}_adv_{adv_acc:.4f}_val_{valid_acc:.4f}_test_{test_acc:.4f}')
            print(f'Saving model [adv_{model_name}_bs{batch_size}_{alfa}_adv_{adv_acc}_val_{valid_acc}_test_{test_acc}] at epoch {epoch + 1}.')
        #elif train_acc > 0.7 and (test_acc > 0.55 or valid_acc > 0.55) and adv_acc == 0:
        #    max_epoch = epoch + 1

        #    model.save(f'{save_path}/tf/{attack_name}/no_adv/adv_{model_name}_bs{batch_size}_{alfa}_train_{train_acc:.4f}_adv_{adv_acc:.4f}_val_{valid_acc:.4f}_test_{test_acc:.4f}')
        #    print(f'Saving model [adv_{model_name}_bs{batch_size}_{alfa}_adv_{adv_acc}_val_{valid_acc}_test_{test_acc}] at epoch {epoch + 1}.')

    print(f'The saved model is [adv_{model_name}_bs{batch_size}_{alfa}] at epoch {max_epoch}.')
    return model