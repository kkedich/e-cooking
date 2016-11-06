
import classifier_from_little_data_script_2 as classifier2
import classifier_from_little_data_script_3 as classifier3
import constants

def main():

    nb_epoch = 50
    input_ingredients = []
    input_train = []
    input_val = []

    # TODO read ingredients from json file and transform in a vector[total_number_of_ingredients]
    # TODO split data in train/validation and test
    # TODO read data

    classifier2.save_bottlebeck_features(constants.file_bottleneck_features_train, constants.file_bottleneck_features_validation,
                                         img_width=constants.IMG_WIDTH, img_height=constants.IMG_HEIGHT,
                                         input_data_train=input_train, input_data_val=input_val,
                                         batch_size=constants.BATCH_SIZE)

    classifier2.train_top_model(constants.file_bottleneck_features_train, constants.file_bottleneck_features_validation,
                                constants.top_model_weights_path,
                                nb_epoch, batch_size=constants.BATCH_SIZE,
                                ingredients=input_ingredients)


    classifier3.fine_tuning(constants.top_model_weights_path,
                            img_width=constants.IMG_WIDTH, img_height=constants.IMG_HEIGHT,
                            batch_size=constants.BATCH_SIZE, nb_epoch=nb_epoch,
                            ingredients=input_ingredients,
                            train_data=input_train)



if __name__ == '__main__':
    main()