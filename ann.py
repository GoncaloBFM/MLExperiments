import numpy
import pandas
import talos
import xgboost
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from seaborn import barplot, lineplot, scatterplot
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GridSearchCV
from keras.callbacks import EarlyStopping, ModelCheckpoint
from talos import Reporting, Evaluate
from talos.model import early_stopper
import matplotlib.pyplot as plt


def basic_processing():
    categorical_features = ['Geography', 'Gender']
    continuous_features = ['CreditScore', 'Tenure', "Age", 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    feature_names = categorical_features + continuous_features
    to_predict = "Exited"
    data = pandas.read_csv("./data/churn.csv")
    data = data.sample(frac=1).reset_index(drop=True)
    scaler = preprocessing.StandardScaler()
    data[continuous_features] = scaler.fit_transform(data[continuous_features])
    x = data[feature_names]
    y = data[to_predict]

    redundant_features = []
    for feature_name in categorical_features:
        encoding = pandas.get_dummies(x[feature_name], prefix=feature_name)
        redundant_features.append(encoding.keys()[0])
        x = pandas.concat([x, encoding], axis=1)
        x.drop([feature_name], axis=1, inplace=True)
    return x, y, redundant_features


def stats():
    # f, ax = plt.subplots(figsize=(10, 6))
    # corr = pandas.concat([X, Y], axis=1).corr()
    # sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
    #             linewidths=.05)
    # f.subplots_adjust(top=0.93)
    # plt.show()
    #
    # germans = X[X["Geography_Germany"] == 1]
    # french = X[X["Geography_France"] == 1]
    # spanish = X[X["Geography_Spain"] == 1]
    # print("avg. german balance", numpy.mean(germans["Balance"]))
    # print("avg. french balance", numpy.mean(french["Balance"]))
    # print("avg. spanish banlance", numpy.mean(spanish["Balance"]))
    #
    # print()
    # print("avg. exited balance", numpy.mean(X["Balance"][Y == 1]))
    # print("std. exited balance", numpy.std(X["Balance"][Y == 1]))
    # print("avg. non exited balance", numpy.mean(X["Balance"][Y == 0]))
    # print("std. non exited balance", numpy.std(X["Balance"][Y == 0]))
    pass


def training_processing(X, Y, redundant_features):
    X.drop(redundant_features, axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)
    return x_train, x_test, y_train, y_test


def do_nn1(x_train, x_test, y_train, y_test):
    print("classifier: NN1")

    def create_nn1(optimizer='adam'):
        seq = Sequential()
        seq.add(Dense(output_dim=12, init="uniform", activation="relu", input_dim=11))
        # nn.add(Dropout(p=0.3))
        seq.add(Dense(output_dim=12, init="uniform", activation="relu"))

        seq.add(Dense(output_dim=12, init="uniform", activation="relu"))
        # nn.add(Dropout(p=0.3))
        # sigmoid output for 1 class | softmax output for multiple classes
        seq.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))
        seq.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])  # vs categorical_crossentropy
        return seq

    # sklearn_nn = KerasClassifier(build_fn=create_nn1, batch_size=10, epochs=100)
    # grid_values = {
    #     "batch_size": [1, 10, 25, 32],
    #     "epochs": [2, 4],
    #     "optimizer": ['adam', 'rmsprop']
    # }
    #
    # grid = GridSearchCV(sklearn_nn, cv=2, param_grid=grid_values)
    # grid.fit(x_train, y_train)
    # print("best validation params", grid.best_params_)
    # print("best validation score", grid.best_score_)

    sklearn_nn = KerasClassifier(build_fn=create_nn1, batch_size=10, epochs=100)
    scores = cross_val_score(sklearn_nn, X=x_train, y=y_train, cv=10, n_jobs=16)
    print(numpy.mean(scores))
    print(numpy.std(scores))

    nn = create_nn1()
    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    callbacks = []
    nn.fit(x_train, y_train, batch_size=10, callbacks=callbacks, epochs=10)
    nn.load_weights('best_model.h5')
    pred = (nn.predict(x_test).reshape(1, -1)[0] > 0.5)
    print("final test score", accuracy_score(pred, y_test))
    print("test size", len(pred))
    confusion_matrix(y_test, pred)


def do_nn2(x_train, x_test, y_train, y_test):
    print("classifier: NN2")

    def create_nn2(optimizer='adam'):
        seq = Sequential()
        seq.add(Dense(output_dim=100, init="uniform", activation="relu", input_dim=11))
        seq.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))
        seq.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])  # vs categorical_crossentropy
        return seq

    sklearn_nn = KerasClassifier(build_fn=create_nn2, batch_size=200, epochs=200)
    scores = cross_val_score(sklearn_nn, X=x_train, y=y_train, cv=10, n_jobs=16)
    print("cross mean score", numpy.mean(scores))
    print("cross std score", numpy.std(scores))


def do_nn2_talos(x_train, x_test, y_train, y_test):
    p = {
        'learning_rate': [0.2, 0.15, .1, 0.05,.05],
        'n_nodes': [90, 100, 110],
        'batch_size': [80, 90, 100, 110, 120]
    }

    def create_nn2_hyperas(x_train, x_test, y_train, y_test, params):
        seq = Sequential()
        seq.add(Dense(output_dim=params['n_nodes'], init="uniform", activation="relu", input_dim=11))
        seq.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))
        seq.compile(optimizer=Adam(lr=params['learning_rate']), loss="binary_crossentropy", metrics=['accuracy'])  # vs categorical_crossentropy

        epochs = 200
        history = seq.fit(x_train, y_train,
                          verbose=False,
                          batch_size=params["batch_size"],
                          epochs=epochs,
                          validation_data=[x_test, y_test],
                          callbacks=[early_stopper(epochs, mode=[0,50])]
                          )

        return history, seq

    result = talos.Scan(x_train.values, y_train.values, x_val=x_test.values, y_val=y_test.values, model=create_nn2_hyperas, params=p)
    return result


def do_sklearn(classifier, x_train, x_test, y_train, y_test):
    print("Classifier: {}".format(str(classifier.__class__)))
    # grid_values = {'max_depth': list(range(1, 11))}
    # grid = GridSearchCV(classifier, cv=10, param_grid=grid_values)
    # grid.fit(x_train, y_train)
    # print("best validation params", grid.best_params_)
    # print("best validation score", grid.best_score_)

    scores = cross_val_score(classifier, X=x_train, y=y_train, cv=10)
    print("cross mean score", numpy.mean(scores))
    print("cross std score", numpy.std(scores))

    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print("final test score", accuracy_score(pred, y_test))
    print("test size", len(pred))
    confusion_matrix(y_test, pred)

result = None

x_train, x_test, y_train, y_test = training_processing(*basic_processing())
pred = result.best_model().predict([x_test.values])>0.5
print("final test score", accuracy_score(pred, y_test))
print("test size", len(pred))
confusion_matrix(y_test, pred)


if False:
    runfile('/home/gbfm/Workspaces/PyCharm/ANN/ann.py', wdir='/home/gbfm/Workspaces/PyCharm/ANN')
    data = training_processing(*basic_processing())

    result = do_nn2_talos(*data)
    do_nn1(*data)
    do_nn2(*data)
    do_sklearn(MLPClassifier(), *data)
    do_sklearn(xgboost.XGBClassifier(), *data)

    report = Reporting(result)
    f = "n_layers learning_rate round_epochs n_nodes batch_size".split(" ")
    history = report.data.sort_values("val_acc", ascending=False)
    y = report.data["val_acc"]
    x = report.data
    lineplot(x[f[4]],y)
    plt.show()

