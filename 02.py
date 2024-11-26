#미니배치 학습
import sys, os
import numpy as np
sys.path.append(os.pardir)
import tensorflow as tf
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from common.gradient import *

#데이터 불러오기,데이터 전처리(정규화 및 원핫인코딩)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, t_train), (x_test, t_test)  = fashion_mnist.load_data()

#데이터 전처리:[0,255]->[0,1],원핫인코딩
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


#테스트 데이터 원핫인코딩:학습 어느정도 되고 나서 적용
t_train_one_hot = tf.keras.utils.to_categorical(t_train, 10)
#t_test_one_hot = tf.keras.utils.to_categorical(t_test, 10)

#print(x_train.shape)
#print(t_train.shape)

#하이퍼 파라메터
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 500
learning_rate = 0.5
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#1에폭당 반복 수
iter_per_epoch =40

for i in range(iters_num):
    #print(i)
    #미니배치 획득
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 계산
    grad = network.gradient(x_batch, t_batch)

    #매개변수 갱신
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭 당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train_one_hot)
        train_acc_list.append(train_acc)
        print("train acc | "+ str(train_acc) +"train loss | "+str(loss))

# 학습 결과 그래프 그리기
plt.plot(train_loss_list, label='train loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(train_acc_list, label='train acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()