import matplotlib.pyplot as plt


def process(history, directory):
    """ Process the history object returned by model.fit.
        For a model trained on a classification problem with a validation dataset,
        this might produce the following listing:
        ['acc', 'loss', 'val_acc', 'val_loss']
        directory: directory where the plots will be saved

        Based on the example of:
        http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        """
    # List all data in history
    print 'History keys: '.format(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(directory + 'history-accuracy.png')

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(directory + 'history-loss.png')