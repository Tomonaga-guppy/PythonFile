#課題1

#課題2
# パーセプトロンは複数の信号を受け取ったときに、一つの信号（0 か 1）を出力するアルゴリズム。
# 各信号に固有のパラメータである重みがあり、ニューロンでは入力された信号に重みを乗算した総和がある閾値を超えると1が出力される（発火という）
# 重みづけした後にNN自身が元から持っている値であるバイアスを値のコントロールの為に足される
# パーセプトロンを重ねて多層パーセプトロンを用いると非線形的な関数により閾値を設定することができる
# ニューラルネットワークとは単純な関数を組み合わせることによってより複雑な関数を表現するもので多層パーセプトロンに比べ
# 基本的にニューラルネットワークの方が性能が高い。
# 入力信号を線形関数で処理した値を活性化関数と呼ばれる非線形関数で処理をする。線形変換と活性化関数の組み合わせをニューロンという

#課題3
# 全結合層とは、ニューラルネットワークにおいて、前の層のすべての出力と次の層のすべての人工ニューロンの入力を相互に接続する層のことです。
# 全結合層は、隠れ層の一種です。

#課題4
# 損失関数（Loss function）とは、「正解値」と、モデルによる出力された「予測値」とのズレの大きさ（これを「Loss：損失」と呼ぶ）を計算するための関数

#課題5
# 逆伝播は「入力層の数 > 出力層」の時、数計算量が少なくなります。

#課題6
# 勾配降下法では、学習するデータセットをいくつかのグループ（バッチと言います）に分けることが一般的です。バッチサイズとは、
# このときに分けられた各グループのことを指します。たとえば、全部で4,000のデータセットがあったとすれば、
# 400ずつに分けたときの400がバッチサイズとなります。

#課題7 省略

#課題8,9
import torch
import numpy as np
# train_images=np.load(r"C:\Users\zutom\.vscode\PythonDataFile\test_images.npy")
train_labels=np.load(r"C:\Users\zutom\.vscode\PythonDataFile\test_images.npy")
# test_images=np.load(r"C:\Users\zutom\.vscode\PythonDataFile\test_images.npy")
# test_labels=np.load(r"C:\Users\zutom\.vscode\PythonDataFile\test_images.npy")

# 課題10
# print("train_images.npyの次元数",train_images.ndim,"train_images.npyの形状",train_images.shape)
# print("test_labels.npyの次元数",test_labels.ndim,"test_labels.npyの形状",test_labels.shape)
# print("test_images.npyの次元数",test_images.ndim,"test_images.npyの形状",test_images.shape)
# print("train_labels.npyの次元数",train_labels.ndim,"train_labels.npyの形状",train_labels.shape)

#課題11
import matplotlib.pyplot as plt
# plt.imshow(train_images[0])
# plt.show()

#課題12
# print("学習用教師データ1個目",train_labels[0])
import cv2
from PIL import Image
#npyファイルをcv2で読み込むために変換
images = Image.fromarray(train_labels[0])
cv2.imshow("image", np.array(images))
cv2.waitKey(0)
exit()

#課題13
train_images=torch.from_numpy(train_images)
train_labels=torch.from_numpy(train_labels)
test_images=torch.from_numpy(test_images)
test_labels=torch.from_numpy(test_labels)

#課題14
train_images_view=train_images.view(60000,784)
test_images_view=test_images.view(10000,784)

#データセットの作成 入力データとラベルをセットに
train_datasets=torch.utils.data.TensorDataset(train_images_view,train_labels)
test_datasets=torch.utils.data.TensorDataset(test_images_view,test_labels)

# inputs=train_datasets[0]
# labels=train_datasets[1]

#課題15、16、17
#参考:https://imagingsolution.net/deep-learning/pytorch/pytorch_mnist_sample_program/

from torch import nn
import torch.nn.functional as F
from torchvision import  transforms
from torch.utils.data import DataLoader

#モデルの作成
# ハイパーパラメータなどの設定値
input_size=28*28        #入力数
hidenlayer_size=196     #隠れ層の数
output_size=10          #出力数
num_epochs = 2         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数
lr = 0.001              # 学習率

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# データローダー
train_dataloader = DataLoader(
    train_datasets,              #第一引数
    batch_size = num_batch,     #バッチサイズ
    shuffle = True)             #ミニバッチの取り出しをランダムにするか

test_dataloader = DataLoader(
    test_datasets,
    batch_size = num_batch,
    shuffle = True)

#ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()                                  #super().__init__()で基底クラス（継承元）のコンストラクタをオーバーライドする
        self.fc1=nn.Linear(input_size,hidenlayer_size)      #28*28の入力数から196個のへ線形変換
        self.fc2=nn.Linear(hidenlayer_size,output_size)
    #順伝播の設定
    def forward(self,x):
        x=self.fc1(x)
        x=torch.relu(x)      #活性化関数の設定 ReLu関数を使用.line94のF.cross_functionでsoftmax関数も一緒に計算
        x=self.fc2(x)
        return F.log_softmax(x,dim=1) #softmax関数計算後、交差エントロピーの為の対数変換

# ニューラルネットワークの生成
model = Net(input_size, output_size).to(device)

#誤差関数の設定 交差エントロピー関数を使用.
error_f=nn.CrossEntropyLoss()

#最適化手法の設定 最急降下法を使用
optimizer=torch.optim.SGD(model.parameters(),lr)

#計算グラフの表示 pip install torchviz, pip install Ipython
from torchviz import make_dot
from IPython.display import display

#グラフの為に空のリスト作成、テンソル化
train_loss_value=[]
test_loss_value=[]

#学習
model.train() #モデルを訓練モードに

for epoch in range(num_epochs): #学習回数分繰り返す
    train_loss_sum =0

    for inputs,labels in train_dataloader:

        #指定したデバイス(GPUかCPU)にデータを送る
        inputs=inputs.to(device)
        labels=labels.to(device)

        ## optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        # inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える(MNISTの場合は必要？)
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        train_loss = error_f(outputs, labels)
        train_loss_sum += train_loss

        # 勾配の計算
        train_loss.backward()

        # 重みの更新
        optimizer.step()

     # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss_sum.item() / len(train_dataloader)}")

    train_loss_value.append(train_loss_sum.item()/len(train_dataloader))

    # モデルの重みの保存
    torch.save(model.state_dict(), 'model_weights.pth')


# 評価
model.eval()  # モデルを評価モードにする

test_loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, input_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        test_loss_sum += error_f(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum()
        #torch.eq(input, other, *, out=None) 第一引数と第二引数が等しいかの真偽を判定する
        #torch.view_as(other) otherと同じサイズにする
        #torch.sum() テンソルの和を取る
        #torch.Tensor.item() 要素の値の取得

print(f"Loss: {test_loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_datasets)}% ({correct}/{len(test_datasets)})")

#グラフの描写
train_loss_value = torch.tensor(train_loss_value).detach()
test_loss_value = torch.tensor(test_loss_value).detach()

plt.figure(figsize=(6,6))
x_range=(1,num_epochs)
plt.plot(x_range,train_loss_value.numpy())
# plt.plot(range(epoch), test_loss_sum, c='#00ff00')
plt.xlim(0, num_epochs)
plt.ylim(0, 0.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.show()
