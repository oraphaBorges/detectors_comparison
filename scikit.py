import sqlite3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
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


def main():
    df = pd.DataFrame(dataset)

    categorize_binary(df)
    # categorize(df)

    y = df.iloc[:, -1]
    X = df.iloc[:, 0:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    algorithms = {
        RandomForestClassifier.__name__: RandomForestClassifier(),
        MLPClassifier.__name__: MLPClassifier(max_iter=1000),
        KNeighborsClassifier.__name__: KNeighborsClassifier(),
        DecisionTreeClassifier.__name__: DecisionTreeClassifier()
    }

    hypers = {
        RandomForestClassifier.__name__: {
            'n_estimators': [2, 4, 8, 16, 32, 64, 128],
            'criterion': ['gini', 'entropy'],
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
        DecisionTreeClassifier.__name__: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]
        }
    }

    for name, alg in algorithms.items():
        print('\nRunning classification with {}...'.format(name))
        gs = GridSearchCV(alg, hypers[name], cv=5, scoring='f1')
        gs.fit(X_train, y_train)
        print('Best parameters for {}:'.format(name), gs.best_params_)
        print('Classification report for {} over test dataset:'.format(name))
        print(classification_report(gs.predict(X_test), y_test))


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
