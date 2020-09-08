from __future__ import division, print_function
from utils import *
from PIL import Image
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

DATA_DIR = "../data"
BATCH_SIZE = 32
NUM_EPOCHS = 20
BEST_MODEL_FILE = os.path.join(DATA_DIR, "models", "test-resnet-best.h5")
FINAL_MODEL_FILE = os.path.join(DATA_DIR, "models", "test-resnet-l1-final.h5")
IMAGE_DIR = os.path.join(DATA_DIR, "photos")
image_triples = get_images(IMAGE_DIR)
triples_data = create_triples(IMAGE_DIR)


def load_image_cache(image_cache, image_filename):
    image = plt.imread(os.path.join(IMAGE_DIR, image_filename))
    image = np.array(Image.fromarray(image).resize((299, 299)))
    image = image.astype("float32")
    image = inception_v3.preprocess_input(image)
    image_cache[image_filename] = image


image_cache = {}
num_pairs = len(triples_data)
for i, (image_filename_l, image_filename_r, _) in enumerate(triples_data):
    if i % 1000 == 0:
        print("images from {:d}/{:d} pairs loaded to cache".format(i, num_pairs))
    if not image_filename_l in image_cache:
        load_image_cache(image_cache, image_filename_l)
    if not image_filename_r in image_cache:
        load_image_cache(image_cache, image_filename_r)
print("images from {:d}/{:d} pairs loaded to cache, COMPLETE".format(i, num_pairs))


def pair_generator(triples, image_cache, datagens, batch_size=32):
    while True:
        indices = np.random.permutation(np.arange(len(triples)))
        num_batches = len(triples) // batch_size
        for bid in range(num_batches):
            batch_indices = indices[bid * batch_size: (bid + 1) * batch_size]
            batch = [triples[i] for i in batch_indices]
            X1 = np.zeros((batch_size, 299, 299, 3))
            X2 = np.zeros((batch_size, 299, 299, 3))
            Y = np.zeros((batch_size, 2))
            for i, (image_filename_l, image_filename_r, label) in enumerate(batch):
                if datagens is None or len(datagens) == 0:
                    X1[i] = image_cache[image_filename_l]
                    X2[i] = image_cache[image_filename_r]
                else:
                    X1[i] = datagens[0].random_transform(image_cache[image_filename_l])
                    X2[i] = datagens[1].random_transform(image_cache[image_filename_r])
                Y[i] = [1, 0] if label == 0 else [0, 1]
            yield [X1, X2], Y


datagen_args = dict(featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    zoom_range=0.2)
datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
pair_gen = pair_generator(triples_data, image_cache, datagens, 32)
inception_1 = inception_v3.InceptionV3(weights="imagenet", include_top=True)
inception_2 = inception_v3.InceptionV3(weights="imagenet", include_top=True)

for layer in inception_1.layers:
    layer.trainable = False
    layer._name = layer.name + "_1"
for layer in inception_2.layers:
    layer.trainable = False
    layer._name = layer.name + "_2"

vector_1 = inception_1.get_layer("avg_pool_1").output
vector_2 = inception_2.get_layer("avg_pool_2").output

sim_head = load_model(os.path.join(DATA_DIR, "models", "resnet50-l1-best.h5"), custom_objects={'LeakyReLU': LeakyReLU})
for layer in sim_head.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

prediction = sim_head([vector_1, vector_2])
model = Model(inputs=[inception_1.input, inception_2.input], outputs=prediction)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
triples_data_trainval, triples_data_test = train_test_split(triples_data, train_size=0.8)
triples_data_train, triples_data_val = train_test_split(triples_data_trainval, train_size=0.9)
datagen_args = dict(rotation_range=10,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2,
                    brightness_range=[0.2, 1.0],
                    zca_whitening=True)
datagens = [ImageDataGenerator(**datagen_args),
            ImageDataGenerator(**datagen_args)]
train_pair_gen = pair_generator(triples_data_train, image_cache, None, BATCH_SIZE)
val_pair_gen = pair_generator(triples_data_val, image_cache, None, BATCH_SIZE)
num_train_steps = len(triples_data_train) // BATCH_SIZE
num_val_steps = len(triples_data_val) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=BEST_MODEL_FILE, save_best_only=True)
history = model.fit(train_pair_gen,
                    steps_per_epoch=num_train_steps,
                    epochs=NUM_EPOCHS,
                    validation_data=val_pair_gen,
                    validation_steps=num_val_steps,
                    callbacks=[checkpoint])


def evaluate_model(model):
    ytest, ytest_ = [], []
    test_pair_gen = pair_generator(triples_data_test, image_cache, None, BATCH_SIZE)
    num_test_steps = len(triples_data_test) // BATCH_SIZE
    curr_test_steps = 0
    for [X1test, X2test], Ytest in test_pair_gen:
        if curr_test_steps > num_test_steps:
            break
        Ytest_ = model.predict([X1test, X2test])
        ytest.extend(np.argmax(Ytest, axis=1).tolist())
        ytest_.extend(np.argmax(Ytest_, axis=1).tolist())
        curr_test_steps += 1
    acc = accuracy_score(ytest, ytest_)
    cm = confusion_matrix(ytest, ytest_)
    cr = classification_report(ytest, ytest_)
    return acc, cm, cr


final_model = load_model(FINAL_MODEL_FILE, custom_objects={'LeakyReLU': LeakyReLU})
acc, cm, cr = evaluate_model(final_model)
print("Accuracy Score: {:.3f}".format(acc))
print(cm)
print(cr)