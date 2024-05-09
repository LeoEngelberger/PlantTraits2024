import os
os.environ["KERAS_BACKEND"] = "torch"
class Config:
    verbose = 1  # Verbosity
    seed = 1  # Random seed
    image_size = [224, 224]  # Input image size
    epochs = 128 # Training epochs
    batch_size = 64 # Batch size
    lr_mode = "cos"  # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_folds = 5  # Number of folds to split the dataset
    fold = 0  # Which fold to set as validation data
    class_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean', ]
    aux_class_names = list(map(lambda x: x.replace("mean", "sd"), class_names))
    num_classes = len(class_names) # Number of classes in the dataset
    aux_num_classes = len(aux_class_names)
    BASE_PATH = "D:\GitHub\PlantTraitsProject\Data\planttraits2024"