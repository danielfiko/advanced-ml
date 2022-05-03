import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.utils import resample

df_train = pd.read_csv("mnist_train.csv", header=None, skiprows=1)
df_test = pd.read_csv("mnist_test.csv", header=None, skiprows=1)
df_train_downsample = resample(df_train,
                               replace=False,
                               n_samples=2000,
                               random_state=42)
df_test_downsample = resample(df_test,
                              replace=False,
                              n_samples=2000,
                              random_state=42)
X_train = df_train_downsample.iloc[:, 1:].copy()
X_test = df_test_downsample.iloc[:, 1:].copy()
y_train = df_train_downsample.iloc[:, :1].copy()
y_test = df_test_downsample.iloc[:, :1].copy()

X_train = X_train
X_test = X_test

# param_grid = [
#  {'C': [0.5, 1, 10, 100], # NOTE: Values for C must be > 0
#   'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
#   'kernel': ['rbf']},
# ]
# optimal_params = GridSearchCV(
#        svm.SVC(),
#        param_grid,
#        cv=5,
#        scoring='accuracy',
#        verbose =0
#   )
# optimal_params.fit(X_train, y_train.values.ravel())
# print(optimal_params.best_params_)
# exit()

#clf = svm.SVC(C=10, gamma="scale", kernel="rbf")
clf = svm.SVC(kernel="linear", C=1)

clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
