from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# 使能 eager execution 模式
tf.enable_eager_execution()

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

# 下载数据集
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
                                           
print("Local copy of the dataset file:{}".format(train_dataset_fp))

# 解析数据集
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]] # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # 头四个字段是特征，合并金一个 tensor 中
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # 最后一个字段是标签
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

# 创建训练 tf.data.Dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1) # 跳过表头行
train_dataset = train_dataset.map(parse_csv) # 解析每一行
train_dataset = train_dataset.shuffle(buffer_size=1000) # 如果样本是随机排列的，则训练效果最好
train_dataset = train_dataset.batch(32)

# 查看一个批次中的单个示例条目
feature, label = iter(train_dataset).next()
print("示例特征：", feature[0])
print("示例标签：", label[0])

# 使用 Keras 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)), # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(3)
])

# 训练模型
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 训练循环

# 保留结果以绘图
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # 训练循环 - 使用 32 批次
    for x, y in train_dataset:
        # 优化模型
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # 跟踪进度
        epoch_loss_avg(loss(model, x, y)) # 添加当前批量损失
        # 比较预测记过和实际结果
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # 结束时代
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}： Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))