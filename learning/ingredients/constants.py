


# Dimensions of our images.
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Dataset variables
TRAIN_DATA_DIR = 'data/train'
VALIDATION_DATA_DIR = 'data/validation'
NB_TRAIN_SAMPLES = 2000 # Considering all the dataset not just only one class
NB_VALIDATION_SAMPLES = 800 # Considering all the dataset not just only one class


# files bottleneck features
file_bottleneck_features_train = 'bottleneck_features_train.npy'
file_bottleneck_features_validation = 'bottleneck_features_validation.npy'
top_model_weights_path = 'bottleneck_fc_model.h5'

BATCH_SIZE = 32
