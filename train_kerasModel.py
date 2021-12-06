
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import regularizers
from keras.layers import Conv2D
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier


#define constant
DIRECTORY = "New_Processed_Images"
FOLDERS = ["with_mask", "without_mask"]
LR = 1e-4
NUM_EPOCH = 20
BATCH_SIZE = 32


# this function plot roc graph
def plot_roc(Xtest, ytest, model, dummy_clf):
    # cnn model
    scores_cnn = model.predict(Xtest, batch_size=BATCH_SIZE)
    fpr, tpr, _ = roc_curve(np.argmax(ytest, axis=1),np.argmax(scores_cnn, axis=1))
    cnn, = plt.plot(fpr, tpr, color='red', label='cnn')

    # Baseline model
    scores_bl = dummy_clf.predict(Xtest)
    fpr, tpr, _ = roc_curve(np.argmax(ytest, axis=1),np.argmax(scores_bl, axis=1))
    baseline, = plt.plot(fpr, tpr, color='blue', label='baseline')

    # Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')

    plt.legend([cnn, baseline], ["cnn", "baseline"])
    plt.show()


#store dataset
print("Storing dataset...")

dataset = []
results = []

for folder in FOLDERS:
    path = os.path.join(DIRECTORY, folder)
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        img = load_img(image_path, target_size=(255, 255))
        img = img_to_array(img)
        img = preprocess_input(img)
		#push to each array
        dataset.append(img)
        results.append(folder)

# sort into one-hot data
lb = LabelBinarizer()
results = lb.fit_transform(results)
results = to_categorical(results)

dataset = np.array(dataset, dtype="float32")
results = np.array(results)

(trainX, testX, trainY, testY) = train_test_split(dataset, results,test_size=0.20, stratify=results, random_state=50)

# Model / data parameters
num_classes = 2
input_shape = (255, 255, 3)

print("orig x_train shape:", trainX.shape)
print("orig y_train shape:", trainY.shape)

#CNN layers
model = keras.Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=trainX.shape[1:],activation='relu'))
model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.001)))  # this L1 value can be changed
model.compile(loss="categorical_crossentropy",optimizer='adam', metrics=["accuracy"])
#--finish CNN


# set variable for model.fit()
opt = Adam(lr=LR, decay=LR / NUM_EPOCH)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
model.summary()
steps_per_epoch = int(np.ceil(trainX.shape[0]//BATCH_SIZE))
History = model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, steps_per_epoch=steps_per_epoch, validation_split=0.1)

# predict using dataset
print("predicting by the use of dataset...")
predications = model.predict(testX, batch_size=BATCH_SIZE)
predications = np.argmax(predications, axis=1)

# save model
print("Saving this CNN model...")
model.save("detect_mask.model", save_format="h5")

# plot the training loss and accuracy
N = len(History.history["loss"])
plt.style.use("bmh")
plt.figure()
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Number of Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("accuracy_graph.png")
plt.show()

# print the accuracy in percentage
test_loss, test_acc = model.evaluate(testX, testY, verbose=2)
print("Validation Accuracy: {}".format(test_acc))
print("Validation Loss: {}".format(test_loss))

val_loss, val_acc = model.evaluate(trainX, trainY, verbose=2)
print("Training Accuracy: {}".format(val_acc))
print("Training Loss: {}".format(val_loss))

# printing classification report
print("Classification Report: ")
print(classification_report(testY.argmax(axis=1), predications,target_names=lb.classes_))

# Baseline model
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(trainX, trainY)
predictions_dummy = dummy_clf.predict(testX)

# printing the confusion matrix
conf_matrix = confusion_matrix(testY.argmax(axis=1), predications)
print("Confusion Matrix of CNN: ")
print(conf_matrix)
print("Confusion Matrix of Baseline: ")
print(confusion_matrix(np.argmax(testY, axis=1),np.argmax(predictions_dummy, axis=1)))

# plot roc-curve
plot_roc(testX, testY, model, dummy_clf)