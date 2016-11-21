
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Dimensions of our images.
# BATCH_SIZE = 32

# Dataset variables
TRAIN_DATA_DIR = 'data/train'
VALIDATION_DATA_DIR = 'data/validation'
NB_TRAIN_SAMPLES = 2000  # Considering all the dataset not just only one class
NB_VALIDATION_SAMPLES = 800  # Considering all the dataset not just only one class


# Files bottleneck features
file_bottleneck_features_train = 'bottleneck_features_train_v2.npy'
file_bottleneck_features_validation = 'bottleneck_features_validation_v2.npy'
# Model files
top_model_weights_path = 'bottleneck_fc_model_v2.h5'
final_vgg16_model = 'vgg16_model_ingredients_v2.h5'


MIN_VALUE = 0.5


# TODO toda vez que usar funcao de custo atualizar esses valores, nao consegui pegar o shape com Tensor.get_shape
NB_INGREDIENTS = 100
NB_INPUT = 0
