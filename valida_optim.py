import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *
from dataset.mnist import load_mnist
from common.util import shuffle_dataset
from common.trainer import Trainer
from common.multi_layer_net_extend import MultiLayerNetExtend

# 데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 하이퍼파라미터 설정
train_size = x_train.shape[0]
batch_size = 256
max_iterations = 100
lr=0.001
activation='relu'


# 검증 데이터 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

# 학습 함수
def __train(weight_decay, dropout_ratio, epochs=50):
    """
    모델 학습 함수
    weight_decay: 가중치 감소율
    dropout_ratio: 드롭아웃 비율
    """
    network1 = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[200, 200, 200, 200, 200],
        output_size=10,
        activation=activation,  # 고정된 활성화 함수
        use_batchnorm=True,
        weight_init_std='he',
        weight_decay_lambda=weight_decay,
        use_dropout=True,
        dropout_ration=dropout_ratio  # 오타 수정: ratio로 통일
    )
    trainer = Trainer(
        network1, x_train, t_train, x_val, t_val,
        epochs=epochs, mini_batch_size=100,
        optimizer='Adam',
        optimizer_param={'lr': lr}, verbose=False  # 고정된 학습률
    )
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    # 탐색할 하이퍼파라미터의 범위 설정
    weight_decay = 10 ** np.random.uniform(-8, -4)  # 가중치 감소
    dropout_ratio = np.random.uniform(0.1, 0.5)  # 드롭아웃 비율 (0.1 ~ 0.5)

    # __train 호출 시 weight_decay와 dropout_ratio만 전달
    val_acc_list, train_acc_list = __train(weight_decay, dropout_ratio)
    print(f"val acc: {val_acc_list[-1]:.4f} | weight_decay: {weight_decay:.6f}, dropout: {dropout_ratio:.2f}")
    key = f"weight_decay: {weight_decay:.6f}, dropout: {dropout_ratio:.2f}"
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    print(f"Best-{i+1} (val acc: {val_acc_list[-1]:.4f}) | {key}")

    plt.subplot(row_num, col_num, i+1)
    plt.title(f"Best-{i+1}")
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list, label='Validation Accuracy')
    plt.plot(x, results_train[key], "--", label='Training Accuracy')
    if i == 0:  # Add legend to the first subplot only
        plt.legend()
    i += 1

    if i >= graph_draw_num:
        break

plt.show()