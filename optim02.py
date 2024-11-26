import tensorflow as tf
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *
from common.util import smooth_curve
from common.multi_layer_net_extend import MultiLayerNetExtend
#배치 정규화 적용
# 데이터셋 불러오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

# 데이터 전처리: [0,255] -> [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten input
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# 테스트 데이터 원핫인코딩: 학습 어느정도 되고 나서 적용
t_train_one_hot = tf.keras.utils.to_categorical(t_train, 10)

train_size = x_train.shape[0]
batch_size = 128
max_iterations =100
lr = 0.01
use_batchnorm=True

# 에폭 계산
iterations_per_epoch = train_size // batch_size
max_epochs = 20

# 옵티마이저 설정: adam 사용, 실험용 설정, 일단 기본 설정으로 적용
optimizers = {}
optimizers['Adam'] = Adam(lr)

networks = {}
train_loss = {}
train_acc = {}
test_loss = {}
test_acc = {}

for key in optimizers.keys():
     #은닉층의 유런수 100개, 5개의 은닉층, 출력층 10개
    networks[key] = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100,100], output_size=10)
    train_loss[key] = []
    train_acc[key] = []
    test_loss[key] = []
    test_acc[key] = []

# 훈련 시작
for epoch in range(max_epochs):
    epoch_loss = {key: [] for key in optimizers.keys()}
    epoch_acc = {key: [] for key in optimizers.keys()}

    for iteration in range(iterations_per_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            acc = networks[key].accuracy(x_batch, t_batch)

            train_loss[key].append(loss)
            train_acc[key].append(acc)

            epoch_loss[key].append(loss)
            epoch_acc[key].append(acc)
    
    # 에폭이 끝날 때 평균 손실 및 정확도 출력
    print(f"========== Epoch {epoch + 1} ==========")
    for key in optimizers.keys():
        avg_loss = np.mean(epoch_loss[key])
        avg_acc = np.mean(epoch_acc[key])
        print(f"{key} | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc:.4f}")

        # 테스트 데이터 손실 및 정확도
        test_loss_epoch = networks[key].loss(x_test, t_test)
        test_acc_epoch = networks[key].accuracy(x_test, t_test)

        test_loss[key].append(test_loss_epoch)
        test_acc[key].append(test_acc_epoch)

        print(f"{key} | Test Loss: {test_loss_epoch:.4f} | Test Accuracy: {test_acc_epoch:.4f}")

# 학습 결과 그래프 그리기
#plt.plot(train_loss['Adam'], label='Train Loss', color='blue')
plt.plot(test_loss['Adam'], label='Test Loss',color='orange')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
#plt.title('Train vs Test Loss')
plt.show()

#plt.plot(train_acc['Adam'], label='Train Accuracy',  color='blue')
plt.plot(test_acc['Adam'], label='Test Accuracy', color='orange')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
#plt.title('Train vs Test Accuracy')
plt.show()

"""
훈련데이터로 실행한 결과:손실은 줄어들고 정확도는 올라가는 것을 확인할 수 있다.
테스트 데이터로 실행한 결과:손실은 줄어들지만 변동성이 큼->과적합..?, 정확도는 점진적으로 증가하나 변동성이 큰것 같음
"""