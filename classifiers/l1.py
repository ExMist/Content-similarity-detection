from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from utils import *

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "photos")
image_triples = get_images(IMAGE_DIR)
NUM_VECTORIZERS = 5
rs = 42
NUM_CLASSIFIERS = 4
scores = np.zeros((NUM_VECTORIZERS, NUM_CLASSIFIERS))
parameters_svc = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
parameters_xgb = {'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 4, 5]}
parameters_rf = {'bootstrap': [True], 'max_depth': [80, 90, 100, 110], 'max_features': [2, 3],
                 'min_samples_leaf': [3, 4, 5], 'min_samples_split': [8, 10, 12], 'n_estimators': [100, 200, 300, 1000]
                 }


def preprocess_data(vector_file, train_size=0.7):
    xdata, ydata = [], []
    vec_dict = load_vectors(vector_file)
    for image_triple in image_triples:
        X1 = vec_dict[image_triple[0].encode()]
        X2 = vec_dict[image_triple[1].encode()]
        xdata.append(np.abs(np.subtract(X1, X2)))
        ydata.append(image_triple[2])
    X, y = np.array(xdata), np.array(ydata)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
    return Xtrain, Xtest, ytrain, ytest


def train_classifier(clf, model, classifier, params=None):
    if params is None:
        best_clf, best_score = cross_validate(Xtrain, ytrain, clf)
        test_report(best_clf, Xtest, ytest)
        save_model(best_clf, get_model_file(DATA_DIR, model, classifier))
    else:
        grd = GridSearchCV(clf, params, cv=10)
        best_clf = grd.fit(Xtrain, ytrain)
        best_score = best_clf.score(Xtest, ytest)
        test_report(best_clf, Xtest, ytest)
        save_model(best_clf.best_estimator_, get_model_file(DATA_DIR, model, classifier))
    return best_score


# VGG16
VECTOR_FILE = os.path.join(DATA_DIR, "vgg16-vectors.tsv")
Xtrain, Xtest, ytrain, ytest = preprocess_data(VECTOR_FILE)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

nb = GaussianNB()
scores[0, 0] = train_classifier(nb, "vgg16", "nb")

svc = svm.SVC(random_state=rs)
scores[0, 1] = train_classifier(svc, "vgg16", "svc", parameters_svc)

xgb = XGBClassifier(random_state=rs)
scores[0, 2] = train_classifier(xgb, "vgg16", "xgb", parameters_xgb)

rf = RandomForestClassifier(random_state=rs)
scores[0, 3] = train_classifier(rf, "vgg16", "xgb", parameters_rf)

# VGG19
VECTOR_FILE = os.path.join(DATA_DIR, "vgg19-vectors.tsv")
Xtrain, Xtest, ytrain, ytest = preprocess_data(VECTOR_FILE)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

nb = GaussianNB()
scores[1, 0] = train_classifier(nb, "vgg19", "nb")

svc = svm.SVC(random_state=rs)
scores[1, 1] = train_classifier(svc, "vgg19", "svc", parameters_svc)

xgb = XGBClassifier(random_state=rs)
scores[1, 2] = train_classifier(xgb, "vgg19", "xgb", parameters_xgb)

rf = RandomForestClassifier(random_state=rs)
scores[1, 3] = train_classifier(rf, "vgg19", "xgb", parameters_rf)

# Res-Net-50
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
Xtrain, Xtest, ytrain, ytest = preprocess_data(VECTOR_FILE)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

nb = GaussianNB()
scores[2, 0] = train_classifier(nb, "resnet50", "nb")

svc = svm.SVC(random_state=rs)
scores[2, 1] = train_classifier(svc, "resnet50", "svc", parameters_svc)

xgb = XGBClassifier(random_state=rs)
scores[2, 2] = train_classifier(xgb, "resnet50", "xgb", parameters_xgb)

rf = RandomForestClassifier(random_state=rs)
scores[2, 3] = train_classifier(rf, "resnet50", "xgb", parameters_rf)

# Iception
VECTOR_FILE = os.path.join(DATA_DIR, "inception-vectors.tsv")
Xtrain, Xtest, ytrain, ytest = preprocess_data(VECTOR_FILE)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

nb = GaussianNB()
scores[3, 0] = train_classifier(nb, "inception", "nb")

svc = svm.SVC(random_state=rs)
scores[3, 1] = train_classifier(svc, "inception", "svc", parameters_svc)

xgb = XGBClassifier(random_state=rs)
scores[3, 2] = train_classifier(xgb, "inception", "xgb", parameters_xgb)

rf = RandomForestClassifier(random_state=rs)
scores[3, 3] = train_classifier(rf, "inception", "xgb", parameters_rf)

# Xception
VECTOR_FILE = os.path.join(DATA_DIR, "xception-vectors.tsv")
Xtrain, Xtest, ytrain, ytest = preprocess_data(VECTOR_FILE)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

nb = GaussianNB()
scores[4, 0] = train_classifier(nb, "xception", "nb")

svc = svm.SVC(random_state=rs)
scores[4, 1] = train_classifier(svc, "xception", "svc", parameters_svc)

xgb = XGBClassifier(random_state=rs)
scores[4, 2] = train_classifier(xgb, "xception", "xgb", parameters_xgb)

rf = RandomForestClassifier(random_state=rs)
scores[4, 3] = train_classifier(rf, "xception", "xgb", parameters_rf)
