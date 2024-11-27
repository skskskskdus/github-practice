import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *
#rom common.util import smooth_curve
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend

#배치 정규화 적용,optim02.py 하이퍼팔라미터 변경
# 데이터셋 불러오기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
"""
텐서플로우 라이브러리 사용할 때 할것

데이터 전처리: [0,255] -> [0,1]
#x_train = x_train.astype('float32') / 255.0
#x_test = x_test.astype('float32') / 255.0

# Flatten input
#x_train = x_train.reshape(x_train.shape[0], -1)
#x_test = x_test.reshape(x_test.shape[0], -1)

# 테스트 데이터 원핫인코딩: 학습 어느정도 되고 나서 적용
#t_train_one_hot = tf.keras.utils.to_categorical(t_train, 10)

"""

train_size = x_train.shape[0]
batch_size = 128
#최대 반복 횟수 100회로 감소
max_iterations =500
#학습률 낮춰서 설정
lr = 0.001
#배치 정규화 적용


# 에폭 계산
#iterations_per_epoch = train_size // batch_size
#최대 에폭 10회로 감소
#max_epochs = 20

# 옵티마이저 설정: adam 사용, 실험용 설정, 일단 기본 설정으로 적용
optimizers = {}
optimizers['Adam'] = Adam(lr)
optimizers['AdaGrad'] = AdaGrad(lr)

networks = {}
train_loss = {}
train_acc = {}
#test_loss = {}
#test_acc = {}

for key in optimizers.keys():
     #은닉층의 유런수 200개, 4개의 은닉층, 출력층 10개
    networks[key] = MultiLayerNetExtend(input_size=784, hidden_size_list=[200,200,200,200], output_size=10,use_batchnorm=True)
    train_loss[key] = []
    train_acc[key] = []
    #test_loss[key] = []
    #test_acc[key] = []

# 훈련 시작
#성능 평가:x_batch, t_batch -> x_test, t_test
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        # 그래디언트 계산 및 가중치 업데이트
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        # 현재 미니배치에 대한 손실 및 정확도 계산
        loss = networks[key].loss(x_batch, t_batch)
        acc = networks[key].accuracy(x_batch, t_batch)

        # 훈련 데이터에 대한 손실 및 정확도 기록
        train_loss[key].append(loss)
        train_acc[key].append(acc)

        # 마지막으로 계산한 손실과 정확도를 저장 (중복 계산 방지)
        last_loss = loss
        last_acc = acc
    
    if i % 100 == 0:
        print(f"========== Epoch {i} ==========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            acc = networks[key].accuracy(x_batch, t_batch)
            # 훈련 데이터 손실 및 정확도 출력
            print(f"{key} | Train Loss: {str(loss)} | Train Accuracy: {str(acc)}")


# 손실 그래프 비교 (옵티마이저별로 그리기)
plt.figure(figsize=(10, 6))
plt.plot(train_loss['Adam'], label='Train Loss (Adam)', color='blue', linestyle='--')
#plt.plot(test_loss['Adam'], label='Test Loss (Adam)', color='red')

plt.plot(train_loss['AdaGrad'], label='Train Loss (AdaGrad)', color='orange', linestyle='--')
#plt.plot(test_loss['AdaGrad'], label='Test Loss (AdaGrad)', color='yellow')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Train vs Test Loss - Adam vs AdaGrad')
plt.legend()
plt.show()

# 정확도 그래프 비교 (옵티마이저별로 그리기)
plt.figure(figsize=(10, 6))
plt.plot(train_acc['Adam'], label='Train Accuracy (Adam)', color='blue', linestyle='--')
#plt.plot(test_acc['Adam'], label='Test Accuracy (Adam)', color='red')

plt.plot(train_acc['AdaGrad'], label='Train Accuracy (AdaGrad)', color='orange', linestyle='--')
#plt.plot(test_acc['AdaGrad'], label='Test Accuracy (AdaGrad)', color='yellow')

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy - Adam vs AdaGrad')
plt.legend()
plt.show()

