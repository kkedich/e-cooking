

IMG_WIDTH, IMG_HEIGHT = 224, 224  # Dimensions of our images.


# Files bottleneck features
file_bottleneck_features_train = 'bottleneck_features_train_v3.npy'
file_bottleneck_features_validation = 'bottleneck_features_validation_v3.npy'


# Model files
top_model_weights_path = 'bottleneck_fc_model_v3.h5'
final_vgg16_model = 'vgg16_model_ingredients_v3.h5'


MIN_VALUE = 0.5 # minimum value in order to define the occurrence of an ingredient. Ex: if x > MIN_VALUE, then
                # we assume that the ingredient which x represents occurs.
