import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *
#rom common.util import smooth_curve
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer
from common.multi_layer_net_extend import MultiLayerNetExtend

#데이터 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

#하이퍼파라미터 설정
train_size = x_train.shape[0]
batch_size = 256
max_iterations =300
#학습률 낮춰서 설정
lr = 0.001

#네트워크 설정1:은닉층 4개,유닛수 200개,배치정규화,가중치 초기화 방법 적용,활성화 함수 relu
#드롭아웃 적용,가중치 감소 적용
network1=MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[200,200,200,200],
    output_size=10,
    activation='relu',
    use_batchnorm=True,
    weight_init_std='he',
     weight_decay_lambda=0.0001,
    use_dropout=True,
    dropout_ration=0.4

)

#네트워크 설정2:은닉층 4개,유닛수 200개,배치정규화,가중치 초기화 방법 적용,활성화 함수 relu
#드롭아웃 적용,가중치 감소 적용
network2=MultiLayerNetExtend(
    input_size=784,
    hidden_size_list=[256,128,256,128],
    output_size=10,
    activation='relu',
    use_batchnorm=True,
    weight_init_std='he',
    weight_decay_lambda=0.0001,
    use_dropout=True,
    dropout_ration=0.3

)

#옵티마이저 설정
optimizer1=Adam(lr)
optimizer2=Adam(lr)

#손실,정확도 저ㅏㅇ
train_loss1,train_acc1,test_acc1,test_loss1=[],[],[],[]
train_loss2,train_acc2,test_acc2,test_loss2=[],[],[],[]

# 학습 루프
for i in range(max_iterations):
    # 배치 샘플링
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 네트워크 1 학습
    grads1 = network1.gradient(x_batch, t_batch)
    optimizer1.update(network1.params, grads1)
    loss1 = network1.loss(x_batch, t_batch)
    acc1 = network1.accuracy(x_batch, t_batch)
    train_loss1.append(loss1)
    train_acc1.append(acc1)

    # 네트워크 2 학습
    grads2 = network2.gradient(x_batch, t_batch)
    optimizer2.update(network2.params, grads2)
    loss2 = network2.loss(x_batch, t_batch)
    acc2 = network2.accuracy(x_batch, t_batch)
    train_loss2.append(loss2)
    train_acc2.append(acc2)

    # 테스트 데이터로 성능 평가
    test_loss01 = network1.loss(x_test, t_test)
    test_acc01 = network1.accuracy(x_test, t_test)
    test_loss02 = network2.loss(x_test, t_test)
    test_acc02 = network2.accuracy(x_test, t_test)

    test_acc1.append(test_acc01)
    test_acc2.append(test_acc02)
    test_loss1.append(test_loss01)
    test_loss2.append(test_loss02)

    # 중간 결과 출력
    if i % 10 == 0:
        print(f"Epoch {i}")
        print(f"Train: Network 1 (Adam) - Loss: {loss1}, Accuracy: {acc1}")
        print(f"Train: Network 2 (Adam) - Loss: {loss2}, Accuracy: {acc2}")

        print(f"Test: Network 1 (Adam) - Loss: {test_loss01}, Accuracy: {test_acc01}")
        print(f"Test:Network 2 (Adam) - Loss: {test_loss02}, Accuracy: {test_acc02}")

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_iterations)
plt.plot(x, train_acc1, marker='o', label='train01', markevery=10, linestyle='--')
plt.plot(x,train_acc2, marker='o', label='train02', markevery=10)
plt.plot(x, test_acc1, marker='s', label='test01', markevery=10, linestyle='--')
plt.plot(x, test_acc2, marker='s', label='test02', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

plt.plot(x, train_loss1,marker='o', label='train01', markevery=10, linestyle='--')
plt.plot(x, train_loss2, marker='o', label='train02', markevery=10)
plt.plot(x, test_loss1,marker='s', label='test01', markevery=10, linestyle='--')
plt.plot(x, test_loss2, marker='o', label='train02', markevery=10)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()