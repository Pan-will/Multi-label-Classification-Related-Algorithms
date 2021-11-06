from numpy import *
from multilabel_hoeffding_tree import MultiLabelHoeffdingTreeClassifier
from molearn.classifiers.BR import BR
from sklearn import linear_model

# random.seed(0)

print('Load data')
import pandas as pd

df = pd.read_csv("data/ohsumed.csv", nrows=1500)
L = 3
N_train = 950

labels = df.columns.values.tolist()[0:L]
data = df.values
T = len(data)
Y = data[:, 0:L]
X = data[:, L:]

print("Experimentation")

h = [BR(h=linear_model.SGDClassifier(n_iter=1)), MultiLabelHoeffdingTreeClassifier()]

from molearn.core.evaluation import prequential_evaluation, get_errors

E_pred, E_time = prequential_evaluation(X, Y, h, N_train)

print("Evaluation")
from molearn.core.metrics import Exact_match, J_index

E = zeros((len(h), T - N_train))
for m in range(len(h)):
    E[m] = get_errors(Y[N_train:], E_pred[m], J=Exact_match)

print("Plot Results")
from matplotlib.pyplot import *

fig, ax = subplots(2)
w = 200
for m in range(len(h)):
    acc = mean(E[m, :])
    time = mean(E_time[m, :])
    print(h[m].__class__.__name__)
    print("Exact Match %3.2f" % mean(acc))
    print("Running Time  %3.2f" % mean(time))
    print("---------------------------------------")
    acc_run = convolve(E[m, :], ones((w,)) / w, 'same')  # [(w-1):]
    ax[0].plot(arange(len(acc_run)), acc_run, "-", label=h[m].__class__.__name__)
    acc_time = convolve(E_time[m, :], ones((w,)) / w, 'same')
    ax[1].plot(arange(len(acc_time)), acc_time, ":", label=h[m].__class__.__name__)

ax[0].set_title("Accuracy (exact match)")
ax[1].set_title("Running Time (ms)")
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
savefig("lab2_fig.pdf")
show()
