import sqlite3

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import \
    ConstantKernel, WhiteKernel, RBF, Matern, \
    RationalQuadratic, ExpSineSquared, DotProduct
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def categorize(df):
    # categorize types that are not numeric
    for label in df.dtypes[df.dtypes == 'object'].index:
        # print(label, df[label].unique())
        df[label] = pd.factorize(df[label])[0]


def categorize_binary(df):
    df.replace(to_replace='Different Object', value=0, inplace=True)
    df.replace(to_replace=r'^Same Object.+', value=1, inplace=True, regex=True)


def main():
    df = pd.DataFrame(dataset)

    categorize_binary(df)
    # categorize(df)

    y = df.iloc[:, -1]
    X = df.iloc[:, 0:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    hypers = {
        SVC.__name__: {
            'C': [1, 0.025],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'probability': [True, False],
            'shrinking': [True, False]
        },
        GaussianNB.__name__: {
        },
        DecisionTreeClassifier.__name__: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]
        },
        MLPClassifier.__name__: {
            'hidden_layer_sizes': [(2,), (8,), (64,), (256,),
                                   (2, 2), (4, 4), (8, 8), (64, 64),
                                   (256, 256),
                                   (2, 4, 2), (64, 256, 64), (256, 32, 2),
                                   (2, 4, 8, 4, 2), (2, 32, 128, 32, 2)],
            'activation': ['logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate_init': [0.5, 0.1, 0.01, 0.001]
        },
        KNeighborsClassifier.__name__: {
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree'],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        },
        GaussianProcessClassifier.__name__: {
            # only kernels
        },
        QDA.__name__: {
        },
        RandomForestClassifier.__name__: {
            'n_estimators': [2, 4, 8, 16, 32, 64, 128],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        },
        AdaBoostClassifier.__name__: {
            'algorithm': ['SAMME', 'SAMME.R']
        }
    }

    """
    Intuitively, precision is the ability of the classifier not to
    label as positive a sample that is negative, and recall is the
    ability of the classifier to find all the positive samples.
    """
    scores = ['accuracy', 'precision', 'recall', 'f1']

    for score in scores:
        print('\n--------------------------------------------------')
        print('Searching for best hyper-params according to {}'.format(score))

        algorithms = {
            SVC.__name__: SVC(),
            GaussianNB.__name__: GaussianNB(),
            DecisionTreeClassifier.__name__: DecisionTreeClassifier(),
            MLPClassifier.__name__: MLPClassifier(max_iter=1000),
            KNeighborsClassifier.__name__: KNeighborsClassifier(),
            GaussianProcessClassifier.__name__: GaussianProcessClassifier(),
            QDA.__name__: QDA(),
            RandomForestClassifier.__name__: RandomForestClassifier(),
            AdaBoostClassifier.__name__: AdaBoostClassifier()
        }

        # a kernel’s hyperparameters are optimized
        # during fitting, so rebuild them here
        hypers[GaussianProcessClassifier.__name__]['kernel'] = \
            [ConstantKernel(), WhiteKernel(), RBF(), Matern(),
             RationalQuadratic(), ExpSineSquared(), DotProduct()]

        for name, alg in algorithms.items():
            print('\n> {}'.format(name))
            gs = GridSearchCV(alg, hypers[name], cv=5, scoring=score)
            gs.fit(X_train, y_train)
            y_hat = gs.predict(X_test)
            print('Best parameters:', gs.best_params_)
            print('Accuracy:', accuracy_score(y_test, y_hat))
            print('Classification report over test dataset:')
            print(classification_report(y_test, y_hat))


if __name__ == '__main__':
    with sqlite3.connect('db.sqlite3') as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC
        """)
        last_table = cursor.fetchone()
        if last_table is not None:
            last_table = last_table[0]
        # tables_cursor.close()
        cursor.execute(
            f"""
            select
                kps_feat_min,
                kps_feat_max, 
                kps_feat_diff_mean, 
                kps_feat_diff_std,
                matches_origin,
                kase 
            from
                {last_table}
            --where
            --    iteration = 2
            --    AND matches is not null
            """)
        dataset = cursor.fetchall()
        print(f'{len(dataset)}-sized dataset from table {last_table}')
        main()
