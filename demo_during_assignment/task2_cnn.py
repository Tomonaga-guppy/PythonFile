import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_images=np.load("C:\\Users\\zutom\\PythonDataFile\\train_images.npy")
train_labels=np.load("C:\\Users\\zutom\\PythonDataFile\\train_labels.npy")
test_images=np.load("C:\\Users\\zutom\\PythonDataFile\\test_images.npy")
test_labels=np.load("C:\\Users\\zutom\\PythonDataFile\\test_labels.npy")

# 画像の形状を変換 最後のランクにチャネル数を追加
train_images=train_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)

#numpyはNWHC形式(バッチサイズ、画像の幅、画像の高さ、チャネル数)で、PytorchはNCWH形式(バッチサイズ、チャネル数、画像の幅、画像の高さ)の順に
#対応しているためランクの入れ替えが必要
train_images=np.transpose(train_images,(0,3,1,2))
test_images=np.transpose(test_images,(0,3,1,2))

train_images=torch.from_numpy(train_images).float()
train_labels=torch.from_numpy(train_labels).long()
test_images=torch.from_numpy(test_images).float()
test_labels=torch.from_numpy(test_labels).long()

train_dataset=torch.utils.data.TensorDataset(train_images,train_labels)
test_dataset=torch.utils.data.TensorDataset(test_images,test_labels)

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ハイパーパラメータなどの設定値
input_size=28
liner_input_size=int((input_size/4)**2*32)      #線形層での入力数
hidenlayer_size=196                             #隠れ層の数
output_size=10                                  #出力数
num_epochs = 2                                  # 学習を繰り返す回数
num_batch = 100                                 # 一度に処理する画像の枚数
# lr = 0.001                                    # 学習率
drop_rate=0.25                                   #ドロップアウトの割合

# データローダー
train_dataloader = DataLoader(
    train_dataset,              #第一引数
    batch_size = num_batch,     #バッチサイズ
    shuffle = True)             #ミニバッチの取り出しをランダムにするか

test_dataloader = DataLoader(
    test_dataset,
    batch_size = num_batch,
    shuffle = True)

#ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self,liner_input_size,output_size):
        super().__init__()                                      #super().__init__()で基底クラス（継承元）のコンストラクタをオーバーライドする
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(drop_rate)                      #25%ユニットをランダムに削除、過学習防止
        self.conv1=nn.Conv2d(1,16,kernel_size=5,padding=2)      #畳み込み1チャネルを16チャネルに、チャネルサイズ5*5
        self.conv2=nn.Conv2d(16,32,kernel_size=5,padding=2)     #畳み込み16チャネルを32チャネルに、チャネルサイズ5*5
        self.maxpool=nn.MaxPool2d(2)                            #2*2でプーリング（2*2の範囲の最大値を取る→
        self.flatten=nn.Flatten()                               #線形変換につかえるように平坦化
        self.fc1=nn.Linear(liner_input_size,hidenlayer_size)    #28*28の入力数から196個のへ線形変換
        self.fc2=nn.Linear(hidenlayer_size,output_size)         #196から10個への出力

    #順伝播の設定
    def forward(self,x):
        x=self.conv1(x)     #(100,1,28,28)-(100,16,28,28) チャネル数、画像の幅、画像の高さ
        x=self.relu(x)
        x=self.maxpool(x)   #-(100,16,14,14)
        # x=self.conv2(x)     #-(100,32,14,14)
        # x=self.relu(x)
        # x=self.maxpool(x)   #-(100,32,7,7)

        x=self.flatten(x)

        x=self.dropout(x)
        x=self.fc1(x)
        x=self.relu(x)                                          #活性化関数の設定 ReLu関数を使用.line94のF.cross_functionでsoftmax関数も一緒に計算
        x=self.fc2(x)
        return F.log_softmax(x,dim=1)                           #softmax関数計算後、交差エントロピーの為の対数変換

# ニューラルネットワークの生成
model = Net(liner_input_size, output_size).to(device)

#誤差関数の設定 交差エントロピー関数を使用.
error_f=nn.CrossEntropyLoss()

#最適化手法の設定 アダム法を使用
optimizer=torch.optim.Adam(model.parameters())

#グラフの為に空のリスト作成
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
        # inputs = inputs.view(-1, liner_input_size) # 画像データ部分を一次元へ並び変える(CNNがあるから不必要)
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = error_f(outputs, labels)
        train_loss_sum += loss

        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {train_loss_sum.item() / len(train_dataloader)}")

    # モデルの重みの保存
    # torch.save(model.state_dict(), 'model_weights.pth')

    # train_loss_value.append(train_loss_sum*num_batch/len(train_dataset))

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
        # inputs = inputs.view(-1, liner_input_size) # 画像データ部分を一次元へ並び変える(CNNがあるから不必要)
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

    # test_loss_value.append(test_loss_sum*num_batch/len(test_dataset)

print(f"Loss: {test_loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")

#グラフの描写
plt.figure(figsize=(6,6))
plt.plot(range(epoch), train_loss_value.detach().numpy())
plt.plot(range(epoch), test_loss_value, c='#00ff00')
plt.xlim(0, epoch)
plt.ylim(0, 0.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()