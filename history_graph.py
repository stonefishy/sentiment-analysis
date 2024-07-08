
import matplotlib.pyplot as plt


def display_training_loss_history(trained_history):
    history_dict = trained_history.history
    history_dict.keys()
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(history_dict['binary_accuracy']) + 1)

    plt.plot(epochs, loss, 'bo', label='Training Loss') # "bo" is for "blue dot"
    plt.plot(epochs, val_loss, 'b', label='Validation Loss') # b is for "solid blue line"
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def display_training_accuracy_history(trained_history):
    history_dict = trained_history.history
    history_dict.keys()
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()