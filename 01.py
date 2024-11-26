import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *
from common.util import smooth_curve 
from common.multi_layer_net import MultiLayerNet

#데이터셋 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, t_train), (x_test, t_test)  = fashion_mnist.load_data()

#데이터 확인,train_images:[0,255],28*28,train_labels:0~9
print(x_train.shape)
print(t_train.shape)

#데이터 전처리:[0,255]->[0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#데이터 전처리 확인
#print(x_train.shape)
#print(x_test.shape)

#데이터 원핫인코딩:손실 함수와의 호환성을 위해
t_train_one_hot = tf.keras.utils.to_categorical(t_train, 10)
t_test_one_hot = tf.keras.utils.to_categorical(t_test, 10)

#print(t_train_one_hot.shape)
#print(t_test_one_hot.shape)

#하이퍼파라미터 설정
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
lr=0.01

#옵티마이저 설정:adam 사용,실험용 설정,일단 기본 설정으로 적용
optimizers = {}
optimizers['Adam'] = Adam(lr)

networks = {}
train_loss={}
train_acc = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []
    train_acc[key] = []

#훈련 시작
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train_one_hot[batch_mask]

    for key in optimizers.keys():
        grads=networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss=networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                accuracy = networks[key].accuracy(x_train, t_train_one_hot)
                train_acc[key].append(accuracy)
                print(key + " - loss: " + str(loss) + ", accuracy: " + str(accuracy))

#그래프 그리기
markers = {"Adam": "D"}
x=np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key],
        markevery=100, label=key)
plt.xlabel("iterations")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()