import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import torch.utils.data
from torch.utils.data import DataLoader
import csv
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from dataloader import Dataset, Dataset_raw
from utils import evaluate_class, validate
import numpy as np
import time

# https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
# correlation between test harness and ideal test condition

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


parser = argparse.ArgumentParser()

parser.add_argument('--fft', type=int, default=4)
parser.add_argument('--stat', type=int, default=1)
parser.add_argument('--MERGE', type=int, default=0)
parser.add_argument('--window_size', type=int, default=30)
parser.add_argument('--split_ratio', type=float, default=0.9)


args = parser.parse_args()
print(f'Training configs: {args}')
name = "ML_merge{}_w{}_f{}".format(args.MERGE, args.window_size, args.fft)
name_merge = "merge{}".format(args.MERGE)
hyper_params = {"fft": args.fft, "stat": args.stat, "MERGE" : args.MERGE,
                "window_size": args.window_size,"split_ratio": args.split_ratio}

result_eval_dict = {"hyper_params": hyper_params}
"""STEP 2: load data"""

df = pd.DataFrame()

df = pd.read_csv("dataset/Telegram_1hour_7.csv")
df.insert(2, "label", int(0))
df_0 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("dataset/Zoom_1hour_5.csv")
df.insert(2, "label", int(1))
df_1 = df[["Time", "Length", "label"]].to_numpy()

df = pd.read_csv("dataset/YouTube_1hour_2.csv")
df.insert(2, "label", int(2))
df_2 = df[["Time", "Length", "label"]].to_numpy()

df_set = np.vstack((df_0, df_1, df_2))

if args.MERGE == 0 or args.MERGE == 7 or args.MERGE == 9:
    df_set = Dataset_raw(df_set, MERGE= args.MERGE)

else:
    df_set = Dataset(df_set, window_size= args.window_size,
                     fft_num= args.fft, stat=args.stat, MERGE= args.MERGE)

train_dataset, val_dataset = torch.utils.data.random_split(
    df_set, [int(len(df_set) *args.split_ratio),
             len(df_set) - int(len(df_set) * args.split_ratio)])

val_dataset, test_dataset = torch.utils.data.random_split(
    val_dataset, [int(len(val_dataset) * args.split_ratio),
                  len(val_dataset) - int(len(val_dataset) * args.split_ratio)])

print("train_dataset:", len(train_dataset))
#print("val_dataset:", len(val_dataset))
print("test_dataset:", len(test_dataset))

"""STEP 3: Make data iterable"""

num_epochs = 2
print("num_epochs:", int(num_epochs))

train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True, num_workers=0)
#val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), drop_last=False, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), drop_last=False, shuffle=True, num_workers=0)


x, y = next(iter(train_loader))

input_dim = x.size()[1]

X_train, y_train = x.squeeze(), y.squeeze()
X_train = X_train.numpy()
y_train = y_train.numpy().astype(int)
print("X_train.shape, y_train.shape:", X_train.shape, y_train.shape)
# x, y = next(iter(val_loader))
# X_val, y_val = x.squeeze(), y.squeeze()
# X_val = X_val.numpy()
# y_val = y_val.numpy()
# print(X_val.shape, y_val.shape)

x, y = next(iter(test_loader))
X_test, y_test = x.squeeze(), y.squeeze()
X_test = X_test.numpy()
y_test = y_test.numpy().astype(int)
print("X_test.shape, y_test.shape:", X_test.shape, y_test.shape)

# get a list of models to evaluate
def get_models():
    models = list()
    # models.append(LogisticRegression())
    # models.append(RidgeClassifier())
    # models.append(SGDClassifier())
    # models.append(PassiveAggressiveClassifier())
    # models.append(KNeighborsClassifier())
    # models.append(DecisionTreeClassifier())
    # models.append(ExtraTreeClassifier())
    # models.append(LinearSVC())
    # models.append(SVC())
    models.append(GaussianNB())
    # models.append(AdaBoostClassifier())
    # models.append(BaggingClassifier())
    # models.append(RandomForestClassifier())
    # models.append(ExtraTreesClassifier())
    # models.append(GaussianProcessClassifier())
    # models.append(GradientBoostingClassifier())
    # models.append(LinearDiscriminantAnalysis())
    # models.append(QuadraticDiscriminantAnalysis())
    return models


def predict_ML(model):
    model.fit(X_train, y_train.astype(int))
    y_pred = model.predict(X_test)
    return y_pred.astype(int)

# define test conditions
# get the list of models to consider
models = get_models()
# evaluate each model
for model in models:
    start = time.time()
    # evaluate model using each test condition
    y_pred = predict_ML(model)

    test_time = "{}".format(time.time()-start)
    print(type(model).__name__ + "time :", test_time, evaluate_class(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
    print(classification_report(y_test, y_pred))

    test_name = type(model).__name__+name
    print(test_name)
    result_test_file = "result/"+name+"/test_"+name_merge
    test_dict = validate(np.asarray(y_test), np.asarray(y_pred), result_test_file)
    test_dict["test time"] = test_time
    test_dict["ML model"] = type(model).__name__

    result_test_dict = {test_name: test_dict}
    result_eval_dict.update(result_test_dict)

result_eval_dict_name = "result/"+name+"/param_eval_"+name

with open(result_eval_dict_name+'.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(result_eval_dict.keys())
    w.writerow(result_eval_dict.values())
