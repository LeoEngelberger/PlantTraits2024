import math
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime

import numpy as np
import keras_core as ks
from keras_core import Input, Model
from matplotlib import pyplot as plt

from LossFunctions import R2Loss, R2Metric
from torch.nn import LeakyReLU

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras_core.src.layers import Dropout, Concatenate, Flatten, Conv2D, MaxPooling1D, Dense, Reshape, Conv3D, \
    MaxPooling3D, MaxPooling2D
from Config import Config
from DataManagment import DataBuilder
#import keras_core as ks


class PlantGuesser():
    def __init__(self):
        self.checkpoint = None
        self.model = None
        self.data_builder = DataBuilder()
        self.build()

    def build(self):
        image_inputs = Input(shape=(3, *Config.image_size), name='images')
        feature_inputs = Input(shape=(len(self.data_builder.FEATURE_COLS),), name='features')
        # Image Processing Layers
        image_net = Conv2D(244, kernel_size=(3, 3), strides=(2, 2), activation="selu")(image_inputs)
        image_net = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="selu")(image_net)
        image_net = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="selu")(image_net)
        image_net = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="selu")(image_net)
        image_net = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="selu")(image_net)
        image_net = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="selu")(image_net)
        image_net = Flatten()(image_net)
        # Metafeature Processing Layers
        feature_net = Dense(576, activation='relu')(feature_inputs)
        feature_net = Dense(326, activation='relu')(feature_net)
        feature_net = Dense(128, activation='relu')(feature_net)

        # Concatenated Layers
        concatenated_net = Concatenate()([image_net, feature_net])

        # Output Layers
        output1 = Dense(128, activation='relu')(concatenated_net)
        output2 = Dense(128, activation='relu')(concatenated_net)
        output1 = Dense(32, activation='selu')(output1)
        output2 = Dense(32, activation='selu')(output2)
        output1 = Dense(16, activation='relu')(output1)
        output2 = Dense(16, activation='relu')(output2)
        output1 = Dense(6, activation='linear')(output1)  # Linear activation for regression
        output2 = Dense(6, activation='linear')(output2)

        output1 = ks.layers.Dense(Config.num_classes, activation=None, name="head")(output1)
        output2 = ks.layers.Dense(Config.aux_num_classes, activation=None, name="aux_head")(output2)

        output = {"head": output1, "aux_head": output2}

        self.model = Model(([image_inputs, feature_inputs]), output)
        self.model.summary()

    def compile(self):
        self.checkpoint = ks.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )
        criterion = {
            "head": R2Loss(use_mask=False),
            "aux_head": R2Loss(use_mask=True),  # use_mask to ignore `NaN` auxiliary labels
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.model.compile(
            optimizer="adam",
            loss={
                "head": R2Loss(use_mask=False),
                "aux_head": R2Loss(use_mask=True),  # use_mask to ignore `NaN` auxiliary labels
            },
            loss_weights={"head": 1.0, "aux_head": 0.2},
            metrics={"head": R2Metric()},
        )

        self.model.summary()

        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

        self.model.fit(
            self.data_builder.train_dataprovider,
            batch_size=Config.batch_size,
            epochs=Config.epochs,
            steps_per_epoch=len(self.data_builder.test_dataframe) // Config.batch_size,
            callbacks=[self.get_lr_scheduler(Config.batch_size, mode=Config.lr_mode,epochs = Config.epochs, plot=True), self.checkpoint],
        )

    def get_lr_scheduler(self, batch_size=8, mode='cos', epochs=10, plot=False):
        lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
        lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

        def lrfn(epoch):  # Learning rate update function
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max
            elif mode == 'exp':
                lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            elif mode == 'step':
                lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
            elif mode == 'cos':
                decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
                phase = math.pi * decay_epoch_index / decay_total_epochs
                lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
            return lr

        if plot:  # Plot lr curve if plot is True
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.title('LR Scheduler')
            plt.show()

        return ks.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

    def examine(self):
        history_dict = self.model.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['learning_rate']
        self.predictions = []
        self.targets = []
        mean_deltas = []

        for batch in self.data_builder.train_dataprovider:
            inpts = batch[0]
            self.targets = batch[1]
            self.predictions = self.model.predict(inpts)
            break

        for i in range(len(self.targets)):
            prediction = self.predictions["head"][i]
            target = self.targets[i].numpy()
            delta = np.abs(prediction - target)  # Calculate absolute differences
            mean_deltas.append(np.mean(delta))

        epochs = range(1, len(loss_values) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        #
        # Plot the loss vs Epochs
        #
        axes[0].plot(epochs, loss_values, 'r', label='Training loss')
        axes[0].plot(epochs, val_loss_values, 'b', label='learning loss')
        axes[0].set_title('Training Loss', fontsize=16)
        axes[0].set_xlabel('Epochs', fontsize=16)
        axes[0].set_ylabel('Loss', fontsize=16)
        axes[0].legend()

        axes[1].plot(mean_deltas, 'r', label='Training loss')
        axes[1].set_title('Mean Delta for batch', fontsize=16)
        axes[1].set_xlabel('Batch', fontsize=16)
        axes[1].set_ylabel('Delta', fontsize=16)
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        self.make_sample_prediction()

    def make_sample_prediction(self):
        # self.model.load_weights("best_model.keras")

        current_time = datetime.now()

        timestamp = current_time.strftime("%Y-%m-%d-%H-%M")  # Modified timestamp format

        filename = f"output-{timestamp}.txt"

        with open(filename, "a") as f:
            for i in range(12):
                prediction = self.predictions["head"][i]
                target = self.targets[i].numpy()

                formatted_predictions = "".join(
                    [
                        ", ".join(
                            f"{name.replace('_mean', '')}: {val:.2f}"
                            for name, val in zip(Config.class_names[j: j + 3], prediction[j: j + 3])
                        )
                        for j in range(0, len(Config.class_names), 3)
                    ]
                )
                formatted_tar = "".join(
                    [
                        ", ".join(
                            f"{name.replace('_mean', '')}: {val:.2f}"
                            for name, val in zip(Config.class_names[j: j + 3], target[j: j + 3])
                        )
                        for j in range(0, len(Config.class_names), 3)
                    ]
                )

                f.writelines('#' * 50 + '\n')
                f.writelines(f"Comparison {i} \n")
                f.writelines('=' * 40 + '\n')
                f.writelines("Predictions: \n")
                f.writelines(formatted_predictions + "\n")
                f.writelines('*' * 40 + '\n')
                f.writelines("Targets: \n")
                f.writelines(formatted_tar + "\n")
                f.flush()

                print('=' * 12)
                print("PREDICTIONS")
                print(formatted_predictions)
                print('*' * 12)
                print("TARGETS")
                print(formatted_tar)

            f.close()
