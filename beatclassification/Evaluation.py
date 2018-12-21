from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import itertools
import numpy as np
import scikitplot as splt
import matplotlib.pyplot as plt

class Evaluation:

    def evaluate( self, predictions, Y_test,classes=None, title=None, one_hot=True, plot=True):
        if classes == None:
            classes = ['N', 'S', 'V', 'F']
        if one_hot:
            if len(classes) > 2:
                predictions = list(map(lambda x: np.argmax(x), predictions))
                Y_test = list(map(lambda x: np.argmax(x), Y_test))
            else:
                predictions = list(map(lambda x : 0 if x <0.5 else 1, predictions))
        if plot:
            cm = confusion_matrix(Y_test, predictions)
            self.plot_confusion_matrix(cm, classes, normalize=False, title=title)
            plt.show()
            plt.close()
            self.plot_confusion_matrix(cm, classes, normalize=True, title='normalized ' +title)
            plt.show()
            plt.close()
        print("average accuracy")
        accuracy = accuracy_score(Y_test, predictions)
        print(accuracy)
        print("per class precision")
        precision = precision_score(Y_test, predictions, average=None)
        print(precision)
        print("average precision")
        av_precision = np.mean(precision)
        print(av_precision)
        print('per class recall')
        recall = recall_score(Y_test, predictions, average=None)
        print(recall)
        print('average recall')
        av_recall = np.mean(recall)
        print(av_recall)
        print('per class f score')
        fscore = f1_score(Y_test, predictions, average=None)
        print(fscore)
        av_fscore = np.mean(fscore)
        print('average fscore')
        print(av_fscore)
        return av_fscore

    def plot_confusion_matrix(self, cm, classes,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[ :, np.newaxis ]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[ 0 ]), range(cm.shape[ 1 ])):
            plt.text(j, i, format(cm[ i, j ], fmt),
                     horizontalalignment="center",
                     color="white" if cm[ i, j ] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
