from json import load
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras.backend as K
from keras.models import load_model

flower_names = {0:'phlox',1:'rose',2:'calendula',3:'iris',4:'leucanthemum maximum',
                    5:'bellflower',6:'viola',7:'rudbeckia laciniata',
                    8:'peony',9:'aquilegia'}

def img_plot(df):
    print("Plot random images of flowers")
    imgs = []
    labels = []
    df = df.sample(frac=1)
    for file, label in zip(df['file'][:25], df['label'][:25]):
        img = cv2.imread("flower_images/flower_images/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        labels.append(label)
    f, ax = plt.subplots(5, 5, figsize=(15,15))
    for i, img in enumerate(imgs):
        ax[i//5, i%5].imshow(img)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title(labels[i])
    plt.show()

def create_datasets(df, img_size):
    imgs = []
    img = 0
    for file in tqdm(df['file']):
        img = cv2.imread("flower_images/flower_images/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size,img_size))
        normalizedimage = cv2.normalize(img, np.zeros((800, 800)), 0, 255, cv2.NORM_MINMAX) # Normalization
        imgs.append(normalizedimage)
    imgs = np.array(imgs)
    df = pd.get_dummies(df['label'])
    return imgs, df

def plot_loss_and_accuracy(model):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(model.history['loss'], label='Training Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Function')
    plt.subplot(2, 2, 2)
    plt.plot(model.history['accuracy'], label='Training Accuracy')
    plt.plot(model.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

def f1_m(precision, recall):
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def print_metrics(model, print_accuracy=False, print_precision=False, print_recall=False, print_f1=False):
    if print_accuracy:
        print("Training accuracy: ", model.history['accuracy'][-1])
        print("Validation accuracy: ", model.history['val_accuracy'][-1])

    if print_precision:
        print("Training precision: ", model.history['precision'][-1])
        print("Validation precision: ", model.history['val_precision'][-1])

    if print_recall:
        print("Training recall: ", model.history['recall'][-1])
        print("Validation recall: ", model.history['val_recall'][-1])

    if print_f1:
        print("Training F1: ", f1_m(model.history['precision'][-1], model.history['recall'][-1]))
        print("Validation F1: ", f1_m(model.history['val_precision'][-1], model.history['val_recall'][-1]))

def save_model(model, location):
    model.save(location)

def test_saved_model(model_location, image_location, img_size):
    model = load_model(model_location)
    imgs = []
    img = cv2.imread(image_location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size,img_size))
    normalizedimage = cv2.normalize(img, np.zeros((800, 800)), 0, 255, cv2.NORM_MINMAX)
    imgs.append(normalizedimage)
    x = np.array(imgs)

    result = model.predict(x)
    index = np.unravel_index(result[0].argmax(), result[0].shape)
    print(flower_names[index[0]])
