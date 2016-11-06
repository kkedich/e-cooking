
import classifier_from_little_data_script_2 as classifier2
import classifier_from_little_data_script_3 as classifier3
import constants as C
from utils import data
from keras import backend as K


def main():
    K.set_image_dim_ordering('th')

    train_path, test_path, data_train, data_test = data.split_data('recipes-ctc.json', '../data/recipes-ctc/', train=0.8)
    input_tensor, input_ingredients = data.load(data_train, train_path, img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT)

    nb_epoch = 50
    validation_split = 0.1  # 10 % of train data for validation

    classifier2.save_bottlebeck_features(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                         img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                                         input_data_train=input_tensor,
                                         batch_size=C.BATCH_SIZE)

    classifier2.train_top_model(C.file_bottleneck_features_train, C.file_bottleneck_features_validation,
                                C.top_model_weights_path,
                                nb_epoch, batch_size=C.BATCH_SIZE, validation_split=validation_split,
                                ingredients=input_ingredients)


    classifier3.fine_tuning(C.top_model_weights_path,
                            img_width=C.IMG_WIDTH, img_height=C.IMG_HEIGHT,
                            batch_size=C.BATCH_SIZE, nb_epoch=nb_epoch,
                            ingredients=input_ingredients,
                            train_data=input_tensor, validation_split=validation_split)


if __name__ == '__main__':
    main()