import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Lambda, LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model, load_model
from utils import *

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "photos")
BATCH_SIZE = 128
NUM_EPOCHS = 20
DROPOUT = 0.2
ALPHA = 0.03
VECTORIZERS = ["InceptionV3", "ResNet"]
MERGE_MODES = ["Concat", "Dot", "l1", "l2"]


def cosine_distance(vecs, normalize=False):
    x = vecs[0]
    y = vecs[1]
    if normalize:
        x = tf.math.l2_normalize(x, axis=0)
        y = tf.math.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)


def cosine_distance_output_shape(shapes):
    return shapes[0]


def euclidean_distance(vecs):
    x = vecs[0]
    y = vecs[1]
    return K.sqrt(K.sum(K.stack([K.square(x), -K.square(y)], axis=1), axis=1))


def euclidean_distance_output_shape(shapes):
    xshape, yshape = shapes
    return xshape


def absdiff(vecs):
    x = vecs[0]
    y = vecs[1]
    return K.abs(K.sum(K.stack([x, -y], axis=1), axis=1))


def absdiff_output_shape(shapes):
    return shapes[0]


def evaluate_model(model_file, test_gen):
    model_name = os.path.basename(model_file)
    model = load_model(model_file, custom_objects={'LeakyReLU': LeakyReLU})
    print("=== Evaluating model: {:s} ===".format(model_name))
    ytrue, ypred = [], []
    num_test_steps = len(test_triples) // BATCH_SIZE
    for i in range(num_test_steps):
        (X1, X2), Y = next(test_gen)
        Y_ = model.predict([X1, X2])
        ytrue.extend(np.argmax(Y, axis=1).tolist())
        ypred.extend(np.argmax(Y_, axis=1).tolist())
    accuracy = accuracy_score(ytrue, ypred)
    print("\nAccuracy: {:.3f}".format(accuracy))
    print("\nConfusion Matrix")
    print(confusion_matrix(ytrue, ypred))
    print("\nClassification Report")
    print(classification_report(ytrue, ypred))
    return accuracy

scores = np.zeros((len(VECTORIZERS), len(MERGE_MODES)))
image_triples = get_images(IMAGE_DIR)

train_triples, val_triples, test_triples = train_test_split(image_triples, splits=[0.7, 0.1, 0.2])
print(len(train_triples), len(val_triples), len(test_triples))

# InceptionV3
VECTOR_SIZE = 2048
VECTOR_FILE = os.path.join(DATA_DIR, "inception-vectors.tsv")
vec_dict = load_vectors(VECTOR_FILE)
train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))

merged = Concatenate(axis=-1)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

pred = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["accuracy"])

best_model_name = get_model_file(DATA_DIR, "inceptionv3", "cat", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)

train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE

history = model.fit(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=NUM_EPOCHS, validation_data=val_gen,
                    validation_steps=val_steps_per_epoch, callbacks=[checkpoint])

final_model_name = get_model_file(DATA_DIR, "inceptionv3", "cat", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 0] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(cosine_distance,
                output_shape=cosine_distance_output_shape)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
best_model_name = get_model_file(DATA_DIR, "inceptionv3", "dot", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=NUM_EPOCHS, validation_data=val_gen,
                              validation_steps=val_steps_per_epoch, callbacks=[checkpoint])
final_model_name = get_model_file(DATA_DIR, "inceptionv3", "dot", "final")
model.save(final_model_name)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 1] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(absdiff, output_shape=absdiff_output_shape)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
best_model_name = get_model_file(DATA_DIR, "inceptionv3", "l1", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])
final_model_name = get_model_file(DATA_DIR, "inceptionv3", "l1", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 2] = best_accuracy if best_accuracy > final_accuracy else final_accuracy
final_model_name = get_model_file(DATA_DIR, "inceptionv3", "l2", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[0, 3] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

VECTOR_SIZE = 2048
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
vec_dict = load_vectors(VECTOR_FILE)
train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Concatenate(axis=-1)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(256, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

fc3 = Dense(128, kernel_initializer="glorot_uniform")(fc2)
fc3 = Dropout(DROPOUT)(fc3)
fc3 = Activation(LeakyReLU(ALPHA))(fc3)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc3)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
best_model_name = get_model_file(DATA_DIR, "resnet50", "cat", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])

final_model_name = get_model_file(DATA_DIR, "resnet50", "cat", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[1, 0] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(cosine_distance,
                  output_shape=cosine_distance_output_shape)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(256, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

fc3 = Dense(128, kernel_initializer="glorot_uniform")(fc2)
fc3 = Dropout(DROPOUT)(fc3)
fc3 = Activation(LeakyReLU(ALPHA))(fc3)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc3)
pred = Activation("softmax")(pred)
model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

best_model_name = get_model_file(DATA_DIR, "resnet50", "dot", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])

final_model_name = get_model_file(DATA_DIR, "resnet50", "dot", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[1, 1] = best_accuracy if best_accuracy > final_accuracy else final_accuracy
train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(absdiff, output_shape=absdiff_output_shape)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(256, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

fc3 = Dense(128, kernel_initializer="glorot_uniform")(fc2)
fc3 = Dropout(DROPOUT)(fc3)
fc3 = Activation(LeakyReLU(ALPHA))(fc3)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc3)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

best_model_name = get_model_file(DATA_DIR, "resnet50", "l1", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])

final_model_name = get_model_file(DATA_DIR, "resnet50", "l1", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[1, 2] = best_accuracy if best_accuracy > final_accuracy else final_accuracy

train_gen = data_generator(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = data_generator(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

input_1 = Input(shape=(VECTOR_SIZE,))
input_2 = Input(shape=(VECTOR_SIZE,))
merged = Lambda(euclidean_distance,
                output_shape=euclidean_distance_output_shape)([input_1, input_2])

fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
fc1 = Dropout(DROPOUT)(fc1)
fc1 = Activation(LeakyReLU(ALPHA))(fc1)

fc2 = Dense(256, kernel_initializer="glorot_uniform")(fc1)
fc2 = Dropout(DROPOUT)(fc2)
fc2 = Activation(LeakyReLU(ALPHA))(fc2)

fc3 = Dense(128, kernel_initializer="glorot_uniform")(fc2)
fc3 = Dropout(DROPOUT)(fc3)
fc3 = Activation(LeakyReLU(ALPHA))(fc3)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc3)
pred = Activation("softmax")(pred)

model = Model(inputs=[input_1, input_2], outputs=pred)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

best_model_name = get_model_file(DATA_DIR, "resnet50", "l2", "best")
checkpoint = ModelCheckpoint(best_model_name, save_best_only=True)
train_steps_per_epoch = len(train_triples) // BATCH_SIZE
val_steps_per_epoch = len(val_triples) // BATCH_SIZE
history = model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen, validation_steps=val_steps_per_epoch,
                              callbacks=[checkpoint])

final_model_name = get_model_file(DATA_DIR, "resnet50", "l2", "final")
model.save(final_model_name)
test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
final_accuracy = evaluate_model(final_model_name, test_gen)

test_gen = data_generator(test_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
best_accuracy = evaluate_model(best_model_name, test_gen)

scores[1, 3] = best_accuracy if best_accuracy > final_accuracy else final_accuracy


merge_strategy = ["Concat", "Dot", "L1", "L2"]

width=0.3
plt.bar(np.arange(scores.shape[1]), scores[0], width, color="#FF7400", label="Xception")
plt.bar(np.arange(scores.shape[1])+width, scores[1], width, color="#269926", label="ResNet-50")
plt.legend(loc=4)
plt.ylabel("accuracy")
plt.xlabel("merge strategy")
plt.xticks(np.arange(scores.shape[1])+0.5*width, merge_strategy,
          rotation=30)
plt.title("Neural Network Classifiers with Image Vectors")
for i in range(len(merge_strategy)):
    print(merge_strategy[i]+" accuracy score with Inception-V3 : {0}        ResNet-50 :{1}      ".format(scores[0][i], scores[1][i]))
