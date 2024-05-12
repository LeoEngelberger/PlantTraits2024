import math
import os

import LossFunctions

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torchvision.models

from datetime import datetime

import numpy as np
import keras as ks
from keras import Input, Model
from matplotlib import pyplot as plt
from torch import nn
from LossFunctions import R2Loss, R2Metric
import CustomActivationFunctions as CAF
import torch
from keras.src.layers import Dropout, Concatenate, Flatten, Conv2D, MaxPooling1D, Dense, Reshape, Conv3D, \
    MaxPooling3D, MaxPooling2D, GlobalAveragePooling2D
from Config import Config
from DataManagment import DataBuilder



class PlantGuesser(nn.Module):
    def __init__(self):
        super(PlantGuesser, self).__init__()
        self.checkpoint = None
        self.data_builder = DataBuilder()
        self.build()

    def build(self):
        #Load backbone
        backbone = torchvision.models.efficientnet_v2_l(torchvision.models.EfficientNet_V2_L_Weights)
        self.backbone_features = backbone.features

        # Image Processing Layers
        self.image_net = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=(3, 3), stride=(2, 2)),  # Adjust input channels accordingly
            nn.SELU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Feature Layers
        self.feature_net = nn.Sequential(
            nn.Linear(len(self.data_builder.FEATURE_COLS), 576),
            nn.ReLU(),
            nn.Linear(576, 326),
            nn.ReLU(),
            nn.Linear(326, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Output Layers
        self.output1 = nn.Sequential(
            nn.Linear(192, 256),  # Adjust input size accordingly
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.conc_output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

        self.output11 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Assuming mapping_to_target_range0to1 returns values between 0 and 1
        )

        self.output12 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Assuming mapping_to_target_range0to1 returns values between 0 and 1
        )

        self.output13 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Assuming mapping_to_target_range0to1 returns values between 0 and 1
        )

        self.output14 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Assuming mapping_to_target_range0to1 returns values between 0 and 1
        )

        self.output15 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Assuming mapping_to_target_range0to1 returns values between 0 and 1
        )

        self.output16 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


        # Define other output layers similarly...

        self.output2 = nn.Sequential(
            nn.Linear(192, 128),  # Adjust input size accordingly
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

    def forward(self, images, features):
        image_features = self.backbone_features(images)
        image_features = self.image_net(image_features)
        image_features = image_features.view(image_features.size(0), -1)

        feature_features = self.feature_net(features)

        concatenated_net = torch.cat((image_features, feature_features), dim=1)

        output1 = self.output1(concatenated_net)
        output11 = self.output11(output1)
        output12 = self.output12(output1)
        output12 = output12*50
        output13 = self.output13(output1)
        output13 = output13*10
        output14 = self.output14(output1)
        output14 = output14*50
        output15 = self.output15(output1)
        output15 = output15*10
        output16 = self.output16(output1)
        output16 = output16*10000
        concatenated_outputs = torch.cat((output11, output12, output13, output14, output15, output16), dim=1)


        output2 = self.output2(concatenated_net)

        return {"head": concatenated_outputs, "aux_head": output2}

    def compile(self):
        self.checkpoint = ks.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_function = LossFunctions.R2Loss()  # Define your loss function accordingly
        self.r2_metric = R2Metric()  # Instantiate R2Metric
        self.to(device)
        self.train_model()

    def train_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate_scheduler = self.get_lr_scheduler(Config.batch_size, mode=Config.lr_mode,epochs = Config.epochs, plot=True)
        self.history = {"loss_per_epoch": [], "val_loss_per_epoch": []}  # Initialize dictionaries
        loss_per_epoch = []
        Config.BatchPerEpoch = len(self.data_builder.train_dataprovider) // Config.batch_size

        for epoch in range(Config.epochs):
            print(f"\n start of epoch {epoch+1}")
            self.train()
            total_loss = 0.0
            self.r2_metric.reset_states()

            # Set the learning rate for this epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lrfn(epoch)

            for batch_idx, (features, labels, aux_labels) in enumerate(self.data_builder.train_dataprovider):
                images, features, labels, aux_labels = features[0].to(device), features[1].to(device), labels.to(
                    device), aux_labels.to(device)
                self.optimizer.zero_grad()
                outputs = self(images, features)
                loss = self.loss_function(outputs["head"], labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.r2_metric.update_state(outputs["head"], labels)
                print(f"batch: {batch_idx+1}/{Config.BatchPerEpoch}, loss: {loss.item():.4f}", end='\r', flush=True)
                if batch_idx > Config.BatchPerEpoch:
                    break
            r2_value = self.r2_metric.result().item()
            print(f"R2 metric: {r2_value}")

            average_loss = total_loss / len(self.data_builder.train_dataprovider)
            self.history["loss_per_epoch"].append(average_loss)
            #print(f"Batch {batch_idx + 1}/{Config.batch_size}, Train Loss: {average_loss:.4f}")

            # Validation step
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                total_val_loss = 0.0
                for batch_idx, (val_features, val_labels, val_aux_labels) in enumerate(self.data_builder.valid_dataset):
                    # Convert validation data to float32
                    val_images, val_features, val_labels, val_aux_labels = val_features[0].to(device), val_features[
                        1].to(device), val_labels.to(device), val_aux_labels.to(device)

                    val_outputs = self(val_images, val_features)
                    val_loss = self.loss_function(val_outputs["head"], val_labels)
                    total_val_loss += val_loss.item()
                    print(f"validation batch: {batch_idx}/{Config.BatchPerEpoch}, loss: {val_loss.item():.4f}", end='', flush=True)
                    if batch_idx > Config.BatchPerEpoch:
                        break
                average_val_loss = total_val_loss / len(self.data_builder.valid_dataset)
                self.history["val_loss_per_epoch"].append(average_val_loss)  # Store validation loss
            self.checkpoint.on_epoch_end(epoch,
                                         logs={"loss": total_loss, "val_loss": average_val_loss})  # Modify val_loss if applicable
            print(f"Epoch {epoch + 1}/{Config.epochs}, Train Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}", end='', flush=True)

    def get_lr_scheduler(self, batch_size=8, mode='cos', epochs=10, plot=True):
        self.lr_start, self.lr_max, self.lr_min = 5e-5, 8e-6 * batch_size, 1e-5
        self.lr_ramp_ep, self.lr_sus_ep, self.lr_decay = 3, 0, 0.75
        if plot:  # Plot lr curve if plot is True
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(epochs), [self.lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
            plt.xlabel('epoch')
            plt.ylabel('lr')
            plt.title('LR Scheduler')
            plt.show()

        return ks.callbacks.LearningRateScheduler(self.lrfn, verbose=False)

    def lrfn(self,epoch):  # Learning rate update function
        if epoch < self.lr_ramp_ep:
            lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * epoch + self.lr_start
        elif epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        elif Config.lr_mode == 'exp':
            lr = (self.lr_max - self.lr_min) * self.lr_decay ** (epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
        elif Config.lr_mode == 'step':
            lr = self.lr_max * self.lr_decay ** ((epoch - self.lr_ramp_ep - self.lr_sus_ep) // 2)
        elif Config.lr_mode == 'cos':
            decay_total_epochs, decay_epoch_index = Config.epochs - self.lr_ramp_ep - self.lr_sus_ep + 3, epoch - self.lr_ramp_ep - self.lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(phase)) + self.lr_min
        return lr

  # Create lr callback

    def examine(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        history_dict = self.history
        loss_values = history_dict['loss_per_epoch']
        val_loss_values = history_dict['val_loss_per_epoch']
        self.predictions = []
        self.targets = []
        mean_deltas = []

        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_val_loss = 0.0
            for batch_idx, (val_features, val_labels, val_aux_labels) in enumerate(self.data_builder.valid_dataset):
                # Convert validation data to float32
                val_images, val_features, val_labels, val_aux_labels = val_features[0].to(device), val_features[
                    1].to(device), val_labels.to(device), val_aux_labels.to(device)
                self.predictions.append(self(val_images, val_features))
                self.targets.append(val_labels)
                print("\nTargets:\n")
                for tensor in self.targets:
                    tensor_list = tensor.tolist()
                    print(f': {tensor_list}')
                val_outputs = self.predictions[batch_idx]
                print("predictions: \n")
                for key, tensor in val_outputs.items():
                    tensor_list = tensor.tolist()
                    print(f'{key}: {tensor_list}')
                val_loss = self.loss_function(val_outputs["head"], val_labels)
                total_val_loss += val_loss.item()
                break

        for i in range(len(self.targets)):
            prediction = self.predictions[0]["head"][i].cpu()
            target = self.targets[i].cpu().numpy()
            delta = np.abs(prediction - target)  # Calculate absolute differences
            mean_deltas.append(delta/len(self.targets[i]))

        epochs = range(1, len(loss_values) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        #
        # Plot the loss vs Epochs
        #
        axes[0].plot(epochs, loss_values, 'r', label='Training loss')
        axes[0].plot(epochs, val_loss_values, 'bo', label='Validation loss')
        axes[0].set_title('Training Loss', fontsize=16)
        axes[0].set_xlabel('Epochs', fontsize=16)
        axes[0].set_ylabel('Loss', fontsize=16)
        axes[0].legend()

        axes[1].plot(mean_deltas[0])
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

        # with open(filename, "a") as f:
        #
        #     for i in range(12):
        #         prediction = self.predictions[0]["head"][i]
        #         target = self.targets[i].numpy()
        #
        #         formatted_predictions = "".join(
        #             [
        #                 ", ".join(
        #                     f"{name.replace('_mean', '')}: {val:.2f}"
        #                     for name, val in zip(Config.class_names[j: j + 3], prediction[j: j + 3])
        #                 )
        #                 for j in range(0, len(Config.class_names), 3)
        #             ]
        #         )
        #         formatted_tar = "".join(
        #             [
        #                 ", ".join(
        #                     f"{name.replace('_mean', '')}: {val:.2f}"
        #                     for name, val in zip(Config.class_names[j: j + 3], target[j: j + 3])
        #                 )
        #                 for j in range(0, len(Config.class_names), 3)
        #             ]
        #         )
        #
        #         f.writelines('#' * 50 + '\n')
        #         f.writelines(f"Comparison {i} \n")
        #         f.writelines('=' * 40 + '\n')
        #         f.writelines("Predictions: \n")
        #         f.writelines(formatted_predictions + "\n")
        #         f.writelines('*' * 40 + '\n')
        #         f.writelines("Targets: \n")
        #         f.writelines(formatted_tar + "\n")
        #         f.flush()
        #
        #         print('=' * 12)
        #         print("PREDICTIONS")
        #         print(formatted_predictions)
        #         print('*' * 12)
        #         print("TARGETS")
        #         print(formatted_tar)
        #
        #     f.close()
