import numpy as np
import pandas as pd
from collections import defaultdict
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import itertools

# Read in dataset
dataset = pd.read_csv('fri_copy.csv', low_memory=False)
dataset['packet_size'] = dataset['Total Length of Fwd Packets'] + dataset['Total Length of Bwd Packets']

# To validate our data, we split the samples into 10 folds, a standard practice for binary classification tasks.
# To get a general sense of how each of the classifiers perform, we iteratively train on 9 of the folds (after 
# finetuning) and test on the one fold that is withheld. We average the performance across the 10 tests to compensate
# for the possibility of overfitting when evaluating our classifiers. 

# sigmoid function for normalizing values between 0 and 1
def sig(x):
    return 1/(1+np.exp(-x))

# classifies the model on training data and returns zero-one loss on test data
def classify(model, x_train, x_test, y_train, y_test):
    classifier = model
    if classifier.__class__.__name__ == "MultinomialNB":
        classifier.fit(sig(x_train),y_train)
    else:
        classifier.fit(x_train,y_train)
    y_predict = classifier.predict(x_test)
    
    # ANALYSIS: 
    print("==================================")
    print(classifier.__class__.__name__ + ":")
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_predict)
    error = zero_one_loss(y_test, y_predict,)
    accuracy = 1 - error
    
    print("Normal Precision: " + str(precision[0]))
    print("Attack Precision: " + str(precision[1]))
    print("Normal Recall: " + str(recall[0])) 
    print("Attack Recall: " + str(recall[1])) 
    print("Normal F1: " + str(f1[0]))
    print("Attack F1: " + str(f1[1]))
    print("Error " + str(error))
    print("Accuracy " + str(accuracy))
    
    # confusion matrix    
    plt.figure()
    classes = ['Normal', 'Attack']
    cm = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # print feature importance
    if classifier.__class__.__name__ == "RandomForestClassifier":
        print("feature importance:" )
        feature_names = ["Length"]
        feat_imp = dict(zip(feature_names, classifier.feature_importances_))
        for feature in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
            print(feature)
    
def run_classification(data, labels): 
    model_error = [0, 0, 0, 0, 0]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)    

    # Evaluate five standard classifiers. 
    classify(KNeighborsClassifier(), x_train, x_test, y_train, y_test)
    classify(RandomForestClassifier(), x_train, x_test, y_train, y_test)
    classify(LinearSVC(), x_train, x_test, y_train, y_test)
    classify(MultinomialNB(), x_train, x_test, y_train, y_test) # A lil weird - performs poorly

    # classify(DecisionTreeClassifier(), x_train, x_test, y_train, y_test) # Doesn't work

def generate_features_and_labels(data):
	features = data.copy(deep=True)
	features['packet_size'] = features['Total Length of Fwd Packets'] + features['Total Length of Bwd Packets']
	labels = features['Label']
	return (features, labels)

dataset_features, dataset_labels = generate_features_and_labels(dataset)

run_classification(dataset_features.as_matrix(), dataset_labels)


## Analysis of Packet Size

# select length data, and split by label
normal_data = dataset[dataset['Label'] == 0]['packet_size']
attack_data = dataset[dataset['Label'] == 1]['packet_size']

# evaluate the histogram
values_a, base_a = np.histogram(attack_data, bins=1000)
values_n, base_n = np.histogram(normal_data, bins=1000)

# evaluate the cumulative distributive function
cumulative_a = np.cumsum(values_a)
cumulative_n = np.cumsum(values_n)

# normalize y
cumulative_a = cumulative_a/max(cumulative_a)
cumulative_n = cumulative_n/max(cumulative_n)

# plot the cumulative function
plt.figure(figsize=(9, 7))   
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)   
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()    
plt.yticks(fontsize=16)    
plt.xticks(fontsize=16)    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")  

attack_line = plt.plot(base_a[:-1], cumulative_a, c='blue', label="Attack Traffic")
normal_line = plt.plot(base_n[:-1], cumulative_n, c='red', label="Normal Traffic")
plt.xlim([0,10000000])
plt.ylim([0,1.01])
plt.xlabel('Packet Size (Bytes)', fontsize=16)
plt.ylabel('CDF', fontsize=16)
plt.title('Packet Size CDF', fontsize=16, y=1.08)
plt.legend(handles=[normal_line[0], attack_line[0]], loc=4, frameon=False, fontsize=16)
plt.savefig('packet_size.png')
