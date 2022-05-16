import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import math
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
import pandas as pd
from xgboost import XGBClassifier

def main():
    #########
    descr = Descriptors._descList
    calc = [x[1] for x in descr]
    #########
    d_smiles = []
    d_names = []
    infile = open("./input/", "r")
    _ = infile.__next__()  # skip header
    for line in infile:
        d_smiles += [line.split("\t")[1].strip()]
        d_names += [line.split("\t")[0]]

    e_smiles = []
    e_names = []
    infile = open("./input/", "r")
    _ = infile.__next__()  # skip header
    for line in infile:
        e_smiles += [line.split("\t")[1].strip()]
        e_names += [line.split("\t")[0].strip()]

    screen_results = np.loadtxt("./input/screening_data.tsv",
                                delimiter='\t',
                                usecols=(2, ),
                                skiprows=1)
    screen_drugs = np.loadtxt("./input/screening_data.tsv",
                            delimiter='\t',
                            usecols=(0, ),
                            skiprows=1,
                            dtype=object)
    screen_excs = np.loadtxt("./input/screening_data.tsv",
                            delimiter='\t',
                            usecols=(1, ),
                            skiprows=1,
                            dtype=object)

    e_x = [
        describe_mol(e, Chem.MolFromSmiles(e_smiles[e]))
        for e in tqdm(range(len(e_smiles)))
    ]
    d_x = [
        describe_mol(d, Chem.MolFromSmiles(d_smiles[d]))
        for d in tqdm(range(len(d_smiles)))
    ]
    d_x_dict = dict(zip(d_names, d_x))
    e_x_dict = dict(zip(e_names, e_x))

    x = []
    y = []
    for j in range(len(screen_results)):
        x += [d_x_dict[screen_drugs[j]] + e_x_dict[screen_excs[j]]]
        y += [screen_results[j]]

    x = np.array(x)
    y = np.array(y)

    # 绘制 ROC 曲线
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import auc
    from sklearn.metrics import plot_roc_curve, roc_curve
    from sklearn.model_selection import StratifiedKFold
    # #############################################################################
    # 获取数据

    # 导入待处理的数据
    n_samples, n_features = x.shape

    X = x

    # #############################################################################
    cv = StratifiedKFold(n_splits=5)

    classifier = XGBClassifier(
        learning_rate=0.05,
        n_estimators=500,  # 树的个数--1000棵树建立xgboost
        max_depth=4,  # 树的深度
        min_child_weight=1,  # 叶子节点最小权重
        gamma=0.1,  # 惩罚项中叶子结点个数前的参数
        # subsample=0.8,  # 随机选择80%样本建立决策树
        # colsample_btree=0.8,  # 随机选择80%特征建立决策树
        objective='binary:logistic',  # 指定损失函数
        # scale_pos_weight=12,  # 解决样本个数不平衡的问题
    )
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    # *********************  使用cmap *****************

    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train],
                    y[train],
                    eval_set=[(X[test], y[test])],
                    early_stopping_rounds=10,
                    eval_metric='auc')
        viz = plot_roc_curve(classifier,
                            X[test],
                            y[test],
                            name='ROC fold {}'.format(i),
                            alpha=0.1,
                            lw=1,
                            color='blue',
                            ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.4)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    ax.plot(mean_fpr,
            mean_tpr,
            color='b',
            label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
            lw=3,
            alpha=.5)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color='blue',
                    alpha=.1,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    # plt.show()
    plt.savefig('./output/Figue1.png', dpi=3000, bbox_inches='tight')

if __name__ == '__name__':
    main()