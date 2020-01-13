import pandas as pd
from keras import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score, \
    accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import KMeansSMOTE
from sklearn.svm import SVC, LinearSVC

# def create_model():
#     model = Sequential()
#     model.add(Dense(9, input_shape=(9,), activation='linear'))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(Dense(110, activation='linear'))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(Dense(80, activation='linear'))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(Dense(20, activation='linear'))
#     model.add(LeakyReLU(alpha=0.1))
#     model.add(Dense(2, activation='softmax'))
#
#     model.compile(Adam(lr=0.003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     print(model.summary())
#     return model


df = pd.read_csv('data/Training Data.csv')
test_df = pd.read_csv('data/Test Data.csv')

print(df.head())
print(df.columns)
print(df.dtypes)

# Removing patient_id since this is auto increment/Unique number.
df.drop(columns='patient_id', inplace=True)

# Converting Categorical features to numeric feature.
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Adherence'] = LabelEncoder().fit_transform(df['Adherence'])

print(df.head(10))

# Checking histogram for data.
# df.hist()
# plt.show()

# Checking pair plot for understanding the date relation ship with target variable.
# sb.pairplot(df, hue='Adherence')
# plt.show()

print(df.corr())
# corr = df.corr()
# sb.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
# plt.show()

# After correlation metrics the strong negative correlation is Prescription_period and very less postive correlation is Tuberculosis, sms reminder.

# Checking any null values are there.
print(df.isnull().values.any())

rob_scaler = RobustScaler()

X = rob_scaler.fit_transform(df.drop(columns='Adherence'))
y = df['Adherence']

x_final_test = rob_scaler.fit_transform(test_df)

# Tried to implement the dimensionality reduction
# pca = PCA(n_components=8)

# X_Pca = pca.fit_transform(X)

# print(X_Pca.n_features)
# print(pca.explained_variance_)

# Checking whether data-set is balanced.
print(df['Adherence'].value_counts(normalize=True) * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# sm = KMeansSMOTE(random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
# clf = SGDClassifier(loss='hinge', max_iter=100, alpha=0.001)
# calibrated_clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',cv=3)  # set the SGD classifier as the base estimator

# grid_params = {'base_estimator__alpha': [0.0001, 0.001,
#                                          0.01]}  # note 'base_estimator__' in the params because you want to change params in the SGDClassifier
# grid_search = GridSearchCV(estimator=calibrated_clf, param_grid=grid_params, cv=3)
# grid_search.fit(X_train, y_train)

# calibrated_clf.fit(X_train, y_train)

# print(grid_search.best_params_)

# predicted = calibrated_clf.predict(X_test)
# prob_pred = model.predict_proba(X_test)
# n_inputs = X_train.shape[1]
# print('model input=', n_inputs)
# model = create_model()
# model.fit(X_train, y_train, validation_split=0.2, batch_size=20, epochs=100, shuffle=True, verbose=2)

# predictions = model.predict_classes(X_test, batch_size=200, verbose=0)

# param_grid = {
#     'epochs': [10, 50, 100],
#     'batch_size': [10, 20, 40, 60, 80, 100]
#     # 'optimizer' :           ['Adam', 'Nadam'],
#     # 'dropout_rate' :        [0.2, 0.3],
#     # 'activation' :          ['relu', 'elu']
# }
#
# clf = KerasClassifier(build_fn=create_model, verbose=0)
#
# grid = GridSearchCV(estimator=clf, param_grid=param_grid)
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# predictions = grid.predict(X_test)
grid_params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

clf = SVC(C=10, gamma=1, kernel='rbf')
# grid_search = GridSearchCV(estimator=clf, param_grid=grid_params, cv=3)
# grid_search.fit(X_train, y_train)
clf.fit(X_train, y_train)
# print(grid_search.best_params_)

predictions = clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))
print(f1_score(y_test, predictions))
print(accuracy_score(y_test, predictions))