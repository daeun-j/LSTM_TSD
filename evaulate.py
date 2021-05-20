import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils import evaluate
import glob
import os
import fnmatch
file_target = "a"
file_label = "b"

A = np.loadtxt('a', delimiter = ',', skiprows = 0, dtype = 'int')
B = np.loadtxt('b', delimiter = ',', skiprows = 0, dtype = 'int')

print(confusion_matrix(A, B, labels=[0, 1, 2]))
print(classification_report(A, B))

evaluate(A, B)
Anew = np.fromfile("result/lr1e-5/test_merge0_lre-5/target.txt")

entries = os.listdir("result")
file_names = []
for entry in entries:
    if fnmatch.fnmatch(file_names, "test*"):
        print(file_names)
    file_names.extend(str(entry))

"LSTM_TSD/result/lr1e-5/test_merge0_lre-5/target.txt"
file_predict = "result/lr1e-5/test_merge0_lre-5/predict.txt"
file_target = "result/lr1e-5/test#_merge0_lre-5/target.txt"
B = np.loadtxt(file_predict, delimiter = ',', skiprows = 0, dtype = 'int')
A = np.loadtxt(file_target, delimiter = ',', skiprows = 0, dtype = 'int')

B = np.loadtxt("./result/lr1e-5/test_merge0_lre-5/target.txt", delimiter = ',', skiprows = 0, dtype = 'int')