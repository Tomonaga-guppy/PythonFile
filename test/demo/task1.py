#課題1
print("Hello World")


X=1
print(X)

Y=[1,2,3]
Y.append (4)
print(Y)


Z = [i for i in range(1,1001)]
print(Z[0],Z[9],Z[403])

print(Z[9:20])

import random
n=random.choice(Z)
if n%2==0:
    print(n,"偶数")
elif n%2==1:
    print(n,"奇数")


import numpy as np
b = np.array([[10, 20], [30, 40],[50,60]])
print(b)

c = np.array([[10, 40], [20, 50],[30,60]])
print(c)

#課題11
print("(1)の次元数",b.ndim,"(1)の形状",b.shape)
print("(2)の次元数",c.ndim,"(1)の形状",b.shape)

#課題12
print("(1)の初めの要素",b[0,0])
print("(2)の初めの要素",c[0,0])

#課題13
print("(1),(2)の要素毎の和\n",b+c,"\n(1),(2)の要素ごとの積\n",b*c)

#課題14
import matplotlib.pyplot as plt

#課題15
A =np.random.normal(loc=0,scale=1,size=100)
B =np.random.normal(loc=1,scale=2,size=100)

fig1,axes1=plt.subplots(1,2)
axes1[0].boxplot(A)
axes1[1].boxplot(B)

plt.show()

#課題16
fig2,axes2=plt.subplots()
x=np.linspace(0,10)
y=x+1
plt.plot(x,y)
plt.show()

#課題17
x1=int(input("x座標1つめ >> "))
y1=int(input("y座標1つめ >> "))
x2=int(input("x座標2つめ >> "))
y2=int(input("y座標2つめ >> "))
d=np.array([x1,y1])
e=np.array([x2,y2])
distance=np.linalg.norm(e-d)
print(distance)

#課題18
import math
n=int(input("正n角形か>> "))
th=math.radians(360/n)
r=1 #単位円周上に頂点を持つ正n角形
f=np.array([1,0])
g=np.array([math.cos(th),math.sin(th)])
distance2=np.linalg.norm(g-f) #n分割された多角形の単位外周長さ
pi=distance2*n/(2*r)
print("円周率= ",pi)