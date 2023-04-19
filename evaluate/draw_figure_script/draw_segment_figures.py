import numpy as np
import pandas as pd
import seaborn as sns

file_ccdeep = r'E:\paper\evaluate_data\evaluation_for_segmentation\measure\CCDeep_segmentation\Classic_Measures.csv'
file_deepcell = r'E:\paper\evaluate_data\evaluation_for_segmentation\measure\deepcell_segmentation\Classic_Measures.csv'
file_cellpose = r'E:\paper\evaluate_data\evaluation_for_segmentation\measure\cellpose_segmentation\Classic_Measures.csv'

tmp = r'E:\paper\evaluate_data\evaluation_for_segmentation\measure\tmp'

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

def handle_data(file_list, key=None):
    data_list = []
    columns = set()
    for file in file_list:
        data = pd.read_csv(file)
        data_list.append(data)
        columns.update(set(data.columns))
    if key not in columns:
        print('keys: ', columns)
        return
    y_ccdeep = data_list[0][[key]]
    y_deepcell = data_list[1][[key]]
    y_cellpose = data_list[2][[key]]
    df = pd.concat([y_ccdeep, y_deepcell, y_cellpose], axis=1)
    df.columns = ['CCDeep', 'DeepCell', 'Cellpose']
    print(df)
    return df


def draw(df):
    df_std = df[['CCDeep', 'DeepCell', 'Cellpose']].std(axis=1)

    # 添加标准差到DataFrame中
    df['Std'] = df_std

    # 绘制条形图
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Category', y='CCDeep', data=df, color='lightblue', ax=ax,
                alpha=0.8, capsize=0.2, errwidth=1, errcolor='black', label='Value1',
                yerr=df_std)
    sns.barplot(x='Category', y='DeepCell', data=df, color='pink', ax=ax,
                alpha=0.8, capsize=0.2, errwidth=1, errcolor='black', label='Value2',
                yerr=df_std)
    sns.barplot(x='Category', y='Cellpose', data=df, color='lightgreen', ax=ax,
                alpha=0.8, capsize=0.2, errwidth=1, errcolor='black', label='Value3',
                yerr=df_std)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Bar Chart with Error Bars', fontsize=14)
    ax.legend()
    plt.show()


# df = handle_data([file_ccdeep, file_deepcell, file_cellpose], key='ACURCY')
#
# df.to_csv(tmp + '\\Accuracy.csv')

def confusion_matrix(TP, FP, TN, FN, title='Confusion Matrix', filename=None):
    # 组织数据为numpy数组
    plt.figure(figsize=(2000,2000), dpi=600)
    confusion_matrix = np.array([[TP, FN], [FP, TN]])
    conf_mat_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    conf_mat_percent = np.round(conf_mat_percent, decimals=4)

    # 设置热力图参数
    labels = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
    categories = ['P', 'N']
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat_percent, annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap='Blues')
    # 添加标签
    ax.set_xlabel('Prediction', fontdict={'weight': 'bold'})
    ax.set_ylabel('Truth', fontdict={'weight': 'bold'})
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    plt.tick_params(axis='x', which='both', length=0)
    plt.tick_params(axis='y', which='both', length=0)
    # ax.set_title(title)
    if filename:
        name = filename
    else:
        name = title
    plt.savefig(r'E:\paper\evaluate_data\evaluation_for_segmentation\figures' + '\\' + name + '.pdf')
    # plt.show()

def get_matrix(TP, FP, TN, FN):
    confusion_matrix = np.array([[TP, FN], [FP, TN]])
    conf_mat_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    conf_mat_percent = np.round(conf_mat_percent, decimals=4)
    return conf_mat_percent

def draw_sub_heatmap(ccdeep, cellpose, deepcell):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=600)
    cm_ccdeep = get_matrix(*ccdeep)
    cm_cellpose = get_matrix(*cellpose)
    cm_deepcell = get_matrix(*deepcell)
    # 在每个子图中绘制热力图
    categories = ['P', 'N']
    sns.heatmap(cm_ccdeep,annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap='Blues', ax=axs[0], cbar=False)
    sns.heatmap(cm_cellpose, annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap='Blues', ax=axs[1], cbar=False)
    sns.heatmap(cm_deepcell,annot=True, fmt='g', xticklabels=categories, yticklabels=categories, cmap='Blues', ax=axs[2])
    for ax in axs:
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)

    axs[0].set_xticks([], minor=True)
    axs[0].set_yticks([], minor=True)

    axs[1].set_xticks([], minor=True)
    axs[1].set_yticks([], minor=True)

    plt.savefig(r'E:\paper\evaluate_data\evaluation_for_segmentation\figures\confusion_matrix.pdf')

    plt.show()

if __name__ == '__main__':
    ccdeep = (175545, 10588, 4002911, 5260)
    cellpose = (156972,	16834,	3996665,	23833)
    deepcell = (82833,	3922,	4009577,	97972)
    # confusion_matrix(175545, 10588, 4002911, 5260)
    # confusion_matrix(*ccdeep, 'CCDeep')
    # confusion_matrix(*cellpose, 'Cellpose')
    # confusion_matrix(*deepcell, 'Deepcell')
    draw_sub_heatmap(ccdeep, cellpose, deepcell)
