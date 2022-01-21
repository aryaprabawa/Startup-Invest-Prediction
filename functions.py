#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc

def show_training_plot(history):
    # To show evolution of accuracy and loss from training and validation set
    # Takes as input the output history from model fit
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Acc')
    plt.plot(history.history['val_accuracy'], label='Validation Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Acc')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid()

def show_confusion_matrix(y_true, y_pred):
    # To show confusion matrix plot
    # Takes as input Keras' data (train/val/test) generator set for true labels and predicted labels
    
    y_pred_classes = (y_pred > 0.5).astype("int32")
    
    plt.figure(figsize = (5,5))

    x_axis_labels = ['Not Invest', 'Invest']
    y_axis_labels = ['Not Invest', 'Invest']

    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix(y_true,y_pred_classes, labels=range(2)),
                annot= True,cbar = False,
                cmap = 'Blues',
                annot_kws={'size' : 15},
                xticklabels=x_axis_labels,
                yticklabels=y_axis_labels)

    plt.xlabel('Predicted Label',fontdict={'size': 16})
    plt.ylabel('True Label' ,fontdict={'size': 16})
    plt.title('Confusion Matrix',fontdict={'size': 16})
    plt.show()
    
def show_roc_plot(y_true, y_pred):
    # To show roc plot with auc metrics
    # Takes as input Keras' data (train/val/test) generator set for true labels and predicted labels
    
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
def print_classification_report(data_generator, y_pred):
    # To print classification report: accuracy, sensitivity, precision, specificity, and F1-score
    # Takes as input Keras' data (train/val/test) generator set for true labels and predicted labels
    
    y_pred_classes = (y_pred > 0.5).astype("int32")
    
    tn, fp, fn, tp = confusion_matrix(data_generator.classes,y_pred_classes, labels=range(2)).ravel()
    
    print('Positive class: Female')
    print('Negative class: Male')
    print('')

    accuracy = (tp+tn)/(tp+fp+fn+tn)
    sensitivity_recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    f1 = 2*(sensitivity_recall * precision) / (sensitivity_recall + precision)

    print('Accuracy:', accuracy)
    print('Sensitivity/Recall:', sensitivity_recall)
    print('Precision:', precision)
    print('Specificity:', specificity)
    print('F1-score:', f1)
