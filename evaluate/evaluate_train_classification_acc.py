import csv
import matplotlib.pyplot as plt

# 读取CSV文件中的数据
# train_steps, train_acces, train_losses, valid_steps, val_acces, val_losses = [], [], [], [], [], []
import numpy as np

with open(r'H:\CCDeep-data\train_classification_detail_20x.csv') as f:
    reader = csv.reader(f)
    train_steps = list(map(int, next(reader)))
    train_acces = list(map(float, next(reader)))
    train_losses = list(map(float, next(reader)))
    valid_steps = list(map(int, next(reader)))
    val_acces = list(map(float, next(reader)))
    val_losses = list(map(float, next(reader)))

# train_data = sorted(zip(train_steps, train_acces, train_losses), key=lambda x: x[0])
# train_steps, train_acces, train_losses = zip(*train_data)
# valid_data = sorted(zip(valid_steps, val_acces, val_losses), key=lambda x: x[0])
# valid_steps, val_acces, val_losses = zip(*valid_data)

# 对训练数据进行均匀采样
train_size = len(train_steps)
valid_size = len(valid_steps)
train_indices = np.linspace(0, train_size - 1, 200, dtype=np.int32)
valid_indices = np.linspace(0, valid_size - 1, 200, dtype=np.int32)
train_steps = [train_steps[i] for i in train_indices]
train_acces = [train_acces[i] for i in train_indices]
train_losses = [train_losses[i] for i in train_indices]

valid_steps = [valid_steps[i] for i in valid_indices]
valid_acces = [val_acces[i] for i in valid_indices]
valid_losses = [val_losses[i] for i in valid_indices]

# 绘制训练精度曲线
# plt.figure(figsize=(10,8))
# plt.plot(train_acces, label='Training Accuracy')
# plt.plot(train_losses, label='Training Loss')
# plt.xlabel('Training Steps')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.show()
#
# # 绘制训练损失曲线
#
# plt.figure(figsize=(10,8))
# plt.plot(val_acces, label='Validation Accuracy')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Valid Steps')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()


import seaborn as sns

# 设置Seaborn样式
sns.set_style('ticks')
# sns.set_style('whitegrid')
sns.set_palette('husl')


# 生成随机数据

def plot_train():
    # 创建图像对象
    # plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # 绘制train acc曲线
    color = sns.color_palette()[0]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Accuracy', color=color)
    ax1.plot(train_acces, color=color, label='Training accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个y轴对象
    ax2 = ax1.twinx()

    # 绘制train loss曲线
    color = sns.color_palette()[1]
    ax2.set_ylabel('Train Loss', color=color)
    ax2.plot(train_losses, color=color, label='Training loss')
    ax2.tick_params(axis='y', labelcolor=color)
    # 设置图像标题
    plt.title('Training Curve')

    plt.savefig(r'H:\CCDeep-data\figure\classification_training\training_curve.pdf')

    # plt.show()

def plot_valid():
    # 创建图像对象
    # plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # 绘制train acc曲线
    color = sns.color_palette()[0]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Valid Accuracy', color=color)
    ax1.plot(valid_acces, color=color, label='Valid accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个y轴对象
    ax2 = ax1.twinx()

    # 绘制train loss曲线
    color = sns.color_palette()[1]
    ax2.set_ylabel('Valid Loss', color=color)
    ax2.plot(valid_losses, color=color, label='Valid loss')
    ax2.tick_params(axis='y', labelcolor=color)
    # 设置图像标题
    plt.title('Valid Curve')
    # plt.show()

    plt.savefig(r'H:\CCDeep-data\figure\classification_training\valid_curve.pdf')

plot_train()