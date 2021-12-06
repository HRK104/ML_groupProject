# import the necessary packages
from tensorflow.keras import regularizers
from keras.layers import Conv2D
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#DIRECTORY = "dataset"
DIRECTORY = "small_dataset"
CATEGORIES = ["with_mask", "without_mask"]


def plot_roc_models(Xtest, ytest, model, dummy_clf):
    # cnn model
    scores = model.predict(Xtest, batch_size=BS)
    #scores = np.argmax(scores, axis=1)
    fpr, tpr, _ = roc_curve(np.argmax(ytest, axis=1),
                            np.argmax(scores, axis=1))
    cnn, = plt.plot(fpr, tpr, color='red', label='cnn')

    # Baseline model
    scores_bl = dummy_clf.predict(Xtest)
    fpr, tpr, _ = roc_curve(np.argmax(ytest, axis=1),
                            np.argmax(scores_bl, axis=1))
    baseline, = plt.plot(fpr, tpr, color='blue', label='baseline')

    # Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')

    plt.legend([cnn, baseline], ["cnn", "baseline"])
    plt.show()


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("Loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=50)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Model / data parameters
num_classes = 2
input_shape = (224, 224, 3)

print("orig x_train shape:", trainX.shape)
print("orig y_train shape:", trainY.shape)

model = keras.Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=trainX.shape[1:],activation='relu'))
model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(
    0.00001)))  # change the weight parameter of L1(bigger and smaller including 0)
model.compile(loss="categorical_crossentropy",
              optimizer='adam', metrics=["accuracy"])

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
#model.build(input_shape=input_shape)
model.summary()
steps_per_epoch = int(np.ceil(trainX.shape[0]//BS))
History = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch, validation_split=0.1)

# make predictions on the testing set
print("Making predictions on the test set...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# serialize the model to disk
print("Saving the current model...")
model.save("detect_mask.model", save_format="h5")

# plot the training loss and accuracy
N = N = len(History.history["loss"])
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
plt.savefig("plot.png")
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
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# Baseline model
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(trainX, trainY)
predictions_dummy = dummy_clf.predict(testX)

# printing the confusion matrix
conf_matrix = confusion_matrix(testY.argmax(axis=1), predIdxs)
print("Confusion Matrix of CNN: ")
print(conf_matrix)
print("Confusion Matrix of Baseline: ")
print(confusion_matrix(np.argmax(testY, axis=1),
      np.argmax(predictions_dummy, axis=1)))

# plot roc-curve
plot_roc_models(testX, testY, model, dummy_clf)
