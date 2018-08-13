import argparse
import sqlite3
import sys

import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import \
    ConstantKernel, WhiteKernel, RBF, Matern, \
    RationalQuadratic, ExpSineSquared, DotProduct
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def categorize(df):
    # categorize types that are not numeric
    for label in df.dtypes[df.dtypes == 'object'].index:
        # print(label, df[label].unique())
        df[label] = pd.factorize(df[label])[0]


def categorize_binary(df):
    df.replace(to_replace='Different Object', value=0, inplace=True)
    df.replace(to_replace=r'^Same Object.+', value=1, inplace=True, regex=True)


def classify(dataset, test_size=0.3, should_categorize=False):
    df = pd.DataFrame(dataset)

    if should_categorize:
        categorize(df)
    else:
        categorize_binary(df)

    y = df.iloc[:, -1]
    X = df.iloc[:, 0:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    hypers = {
        AdaBoostClassifier.__name__: {
            'algorithm': ['SAMME', 'SAMME.R']
        },
        BernoulliNB.__name__: {
            'fit_prior': [True, False]
        },
        DecisionTreeClassifier.__name__: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]
        },
        GaussianNB.__name__: {
        },
        GaussianProcessClassifier.__name__: {
            # only kernels
        },
        KNeighborsClassifier.__name__: {
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree'],
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        },
        MLPClassifier.__name__: {
            'hidden_layer_sizes': [(2,), (64,),
                                   (2, 2), (4, 4),
                                   (2, 4, 2), (64, 8, 2),
                                   (2, 4, 8, 4, 2)],
            'activation': ['logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate_init': [0.5, 0.1, 0.01, 0.001]
        },
        MultinomialNB.__name__: {
            'fit_prior': [True, False]
        },
        QDA.__name__: {
        },
        RandomForestClassifier.__name__: {
            'n_estimators': [2, 4, 8, 16, 32, 64],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        },
        SGDClassifier.__name__: {
            'loss': ['hinge', 'log', 'modified_huber',
                     'squared_hinge', 'perceptron'],
            'penalty': ['none', 'l1', 'l2', 'elasticnet'],
            'alpha': [0.1, 0.01, 0.001, 0.0001],
            'fit_intercept': [True, False]
        }
    }

    """
    "Intuitively, precision is the ability of the classifier not to
    label as positive a sample that is negative, and recall is the
    ability of the classifier to find all the positive samples."
        scikit-learn docs.
    """
    scores = ['accuracy', 'precision', 'recall', 'f1']

    for score in scores:
        print('\n--------------------------------------------------')
        print(f'Searching for best hyper-params according to {score}')

        algorithms = {
            AdaBoostClassifier.__name__: AdaBoostClassifier(),
            BernoulliNB.__name__: BernoulliNB(),
            DecisionTreeClassifier.__name__: DecisionTreeClassifier(),
            GaussianNB.__name__: GaussianNB(),
            GaussianProcessClassifier.__name__: GaussianProcessClassifier(),
            KNeighborsClassifier.__name__: KNeighborsClassifier(),
            MLPClassifier.__name__: MLPClassifier(max_iter=1000),
            MultinomialNB.__name__: MultinomialNB(),
            QDA.__name__: QDA(),
            RandomForestClassifier.__name__: RandomForestClassifier(),
            SGDClassifier.__name__: SGDClassifier()
        }

        # kernels hyperparameters are optimized
        # during fitting, so we reset them here
        hypers[GaussianProcessClassifier.__name__]['kernel'] = \
            [ConstantKernel(), WhiteKernel(), RBF(), Matern(),
             RationalQuadratic(), ExpSineSquared(), DotProduct()]

        for name, alg in algorithms.items():
            print(f'\n> {name}')
            gs = GridSearchCV(alg, hypers[name], cv=5, scoring=score)
            gs.fit(X_train, y_train)
            y_hat = gs.predict(X_test)
            print('Best parameters:', gs.best_params_)
            print('Accuracy:', accuracy_score(y_test, y_hat))
            print('Classification report over test dataset:')
            print(classification_report(y_test, y_hat))


def main():
    with sqlite3.connect(args.database) as conn:
        cursor = conn.cursor()
        table_name = args.table
        if table_name is None:
            cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' ORDER BY name DESC
            """)
            table_name = cursor.fetchone()
            if table_name is None:
                sys.exit(
                    f'Could not find any tables in database {args.database}')
            table_name = table_name[0]
        query = f"""
            select
                kps_feat_min,
                kps_feat_max, 
                kps_feat_diff_mean, 
                kps_feat_diff_std,
                matches_origin,
                kase 
            from
                {table_name} -- yep, SQL injection issue here \o/
            where
                name = ?
            """
        cursor.execute(query, (args.algorithm,))
        dataset = cursor.fetchall()
        print('Arguments:', args)
        print(f'{len(dataset)}-sized dataset')
        classify(dataset, test_size=args.test_size,
                 should_categorize=args.categorize)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('algorithm',
                      type=str,
                      help='one of: AKAZE, BRISK, ORB, SIFT, SURF'
                           ' (case-sensitive)')
    argp.add_argument('-d', '--database', type=str, default='db.sqlite3',
                      help='SQLite3 database path. Default = %(default)s')
    argp.add_argument('-t', '--table', type=str, default=None,
                      help='SQLite3 database table. Default = last one in'
                           ' reverse lexicographic order of the names'
                           ' (good for table names with timestamps)')
    argp.add_argument('-s', '--test-size', type=float, default=0.3,
                      help='0.0 < test size < 1.0. Default = %(default)s')
    argp.add_argument('-c', '--categorize', action='store_true',
                      help='categorize labels instead of \'binarize\'')
    args = argp.parse_args()

    main()
