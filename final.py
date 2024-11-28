import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# 替换为你的 .set 文件路径
file_path = '241.set'

# 使用 mne.io.read_epochs_eeglab 读取数据
epochs = mne.io.read_epochs_eeglab(file_path)
epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
events = epochs.events
event_id = epochs.event_id

# 打印事件 ID 和标签
print("Event IDs and labels:", event_id)

# 函数：从事件 ID 中提取类别
def extract_event_category(event_id_str):
    numbers = [int(num) for num in event_id_str.split('/') if num.isdigit()]
    for num in [5, 6, 7, 8]:
        if num in numbers:
            return num
    return None  # 如果没有找到，则返回 None

# 将每个事件 ID 映射到类别
event_categories = {event_id_str: extract_event_category(event_id_str) for event_id_str in event_id.keys()}

# 打印事件 ID 到类别的映射
print("Event ID to category mapping:")
for event_id_str, category in event_categories.items():
    print(f"Event ID: {event_id_str}, Category: {category}")

# 提取事件的标签
event_labels = [event[2] for event in events]  # 事件标签

# 从 event_id 字典中反向查找 ID 字符串
id_to_category = {v: extract_event_category(k) for k, v in event_id.items()}
print("ID to category mapping:")
print(id_to_category)

# 获取每个事件的类别
event_categories_labels = [id_to_category.get(label, None) for label in event_labels]

# 打印提取的事件类别和标签
print("Event categories based on labels:")
print(event_categories_labels)

# 统计每个类别的数量
category_counts = Counter(event_categories_labels)
print("Category counts:")
print(category_counts)

# 过滤出有效的类别
valid_categories = [cat for cat in event_categories_labels if cat is not None]

# 打印有效的类别
print("Valid categories:")
print(valid_categories)

# Step 2: Prepare data for classification
selected_labels = [5, 6, 7, 8]
selected_epochs = np.isin(event_categories_labels, selected_labels)
epochs_data = epochs_data[selected_epochs]
labels = np.array([label for label in event_categories_labels if label in selected_labels])

# 将标签映射到 0 到 3
label_mapping = {5: 0, 6: 1, 7: 2, 8: 3}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # 反向映射
labels = np.array([label_mapping[label] for label in labels])

# Step 3: Iterate over time points and save RDM heatmaps
sampling_rate = int(epochs.info['sfreq'])  # 获取采样率
time_interval = 100  # 每 100ms 输出一张热图
num_conditions = len(selected_labels)  # 4 个条件
output_dir = 'rdm_heatmaps'  # 保存图片的目录
os.makedirs(output_dir, exist_ok=True)

# 原始标签列表
condition_labels = [reverse_label_mapping[i] for i in range(num_conditions)]

# 获取时间点数组，单位为秒
times = epochs.times * 1000  # 转换为毫秒
time_points = np.arange(-1000, 2001, time_interval)  # 从 -1000ms 到 2000ms，步长为 100ms

# 遍历每100ms的时间点
for time_point in time_points:
    if time_point < times[0] or time_point > times[-1]:
        continue

    # 寻找最接近的时间点索引
    idx = np.argmin(np.abs(times - time_point))

    X = epochs_data[:, :, idx]  # 选择特定时间点的数据

    # 初始化 RDM
    rdm = np.zeros((num_conditions, num_conditions))

    # 计算每对条件的解码准确率
    for i in range(num_conditions):
        for j in range(i + 1, num_conditions):
            accuracies = []  # 用于存储多次计算的准确率

            # 进行100次重复计算
            for _ in range(100):
                # 提取条件 i 和 j 的数据
                condition_i = X[labels == i]
                condition_j = X[labels == j]

                # 创建 SVM 分类器
                svm = SVC(kernel='linear')

                # 为二分类准备标签
                y_i = np.ones(condition_i.shape[0])  # 条件 i 的标签
                y_j = np.zeros(condition_j.shape[0])  # 条件 j 的标签
                X_binary = np.concatenate([condition_i, condition_j])
                y_binary = np.concatenate([y_i, y_j])

                # 计算 SVM 准确率，使用 10 折交叉验证
                accuracy = cross_val_score(svm, X_binary, y_binary, cv=10).mean()
                accuracies.append(accuracy)

            # 计算 100 次重复的平均准确率
            rdm[i, j] = rdm[j, i] = 1 - np.mean(accuracies)  # 使用不相似性度量

    # 绘制 RDM 热图
    mask = np.triu(np.ones_like(rdm, dtype=bool))  # 创建上三角的掩码
    plt.figure(figsize=(8, 6))
    sns.heatmap(rdm, cmap='RdBu_r', annot=True, fmt='.2f', cbar=True, mask=mask, vmin=0, vmax=1,
                xticklabels=condition_labels, yticklabels=condition_labels)
    plt.title(f'4x4 RDM at Time Point {int(time_point)}ms')
    plt.xlabel('Condition')
    plt.ylabel('Condition')

    # 保存热图
    file_name = os.path.join(output_dir, f'rdm_{int(time_point)}ms.png')
    plt.savefig(file_name)
    plt.close()
    print(f'Saved heatmap for time point {int(time_point)}ms as {file_name}')
