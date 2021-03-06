# 图像去噪

[TOC]

## 1. 噪声的描述

一般使用统计意义上的均值以及方差描述噪声。均值表明了图像中噪声分布的总体强度，方差表明了图像中噪声分布的强弱差异.

### 1.1 噪声模型

#### 高斯噪声

+ 电子电路噪声
+ 高温导致的传感器噪声
+ 暗光导致的传感器噪声

噪声的概率密度函数：
$$
p(z) = \frac{1}{\sqrt{2\pi}\sigma}e^{-(z-  u)/2\sigma ^2}
$$

#### 瑞利噪声

适用于深度图像，不对称的直方图。其概率密度函数:
$$
p(z) = \left\{
\begin{aligned}
\frac{2}{b}(z-a)e^{-(z-a)^2/b} & & z \ge a \\
0 & & z < a
\end{aligned}
\right.
$$
概率密度的均值以及方差:
$$
\mu = a + \sqrt{\pi b /4} \\ \sigma^2 = \frac{b(4-\pi)}{4}
$$

#### 伽马噪声

概率密度曲线为Gamma曲线:

![Gamma](http://media.innohub.top/190524-gamma.png)

#### 指数噪声

适用于激光成像的的图像。其概率密度曲线为指数函数：

#### 均匀噪声

均匀分布

#### 脉冲噪声

即椒盐噪声，出现在随机位置早点深度基本固定的噪声。

![noise](http://media.innohub.top/190524-noise.PNG)

## 2. 去噪

### 2.1 均值滤波

+ 算数均值滤波： 减少噪声的同时模糊了图像信息。**使用邻域内的像素的均值作为滤波后的结果**

+ 几何均值滤波： 达到与算数均值相当的平滑度，但是丢失更少的图像细节。**使用邻域内像素值的累积的开方作为滤波结果**
  $$
  \hat{f}(x, y) = [\prod_{(s, t) \in S} g(s,t) ] ^{1/mn}
  $$

+ 谐波滤波：对于盐噪声效果较好，不适用于椒噪声，善于处理高斯噪声等

  邻域的大小除以像素值倒数的累加和。

+ 逆谐波滤波：适用于椒盐噪声，但不可同时去除

  邻域像素Q+1次方的的累加和与像素Q次方累加和的比值

  Q > 0 时消除椒噪声；Q<0是消除盐噪声；Q=0是算数均值滤波；Q=-1为谐波均值滤波
  $$
  \hat{f}(x, y) = \frac{\sum _{(s,t) \in S} g(s, t) ^{Q+1}}{\sum _{(s,t) \in S} g(s, t) ^{Q}}
  $$

### 2.2 统计排序

+ 中值滤波器： 对于多种噪声具有良好的去噪能力，引起的模糊较小，对于脉冲噪声特别有效

+ 最大最小值滤波器：最大值发现亮点，消除椒噪声；最小值发现暗点，消除盐噪声

+ 中点滤波器：最大值以及最小值的均值，结合了统计排序以及求均值的操作，使其对于高斯、均匀分布的噪声有较好的效果

+ alpha裁剪均值滤波
  $$
  \hat{f}(x, y) = \frac{1}{mn - d}\sum _{(s, t) \in S} g(s, t)
  $$
  d可以取值0~mn-1:

  + d = 0: 均值滤波
  + d = mn - 1: 中值滤波

经过均值滤波，脉冲噪声被除去，震荡部分被平滑，斜坡以及阶跃被保存下来

### 2.3 自适应的局部噪声去除

自适应滤波器可以根据自身状态以及环境调整自身参数以及预先设定的目标。自适应滤波器是基于Mxn的矩形窗口设计的，对于一个局部区域，具有以下四个量:

+ g(x, y) 在(x, y)点处的灰度值
+ $\sigma^2$ 全部噪声方差
+ $m_L$局部区域的局部均值
+ $\sigma ^ 2 _L$ 局部方差

滤波器的预期性能如下:

+ 如果全局方差为0，则滤波器应简单的返回g(x, y)
+ 如果局部方差远远大于全局方差，应返回一个g(x, y)的近似值
+ 如果两个方差相等，返回局部区域的算数均值

$$
\hat{f}(x, y) = g(x, y) - \frac{\sigma ^2}{\sigma_L ^2}[g(x, y) - m_L]
$$

**只需要得到全局噪声方差**

### 2.4 自适应中值滤波

中值滤波的尺寸太小，可以较好地保护图像的某些细节，但是往往遗漏噪声，多次滤波造成特征丧失、图像模糊；如果尺寸太大可以更好抑制噪声，但是图像会很模糊。可以先采用小尺寸的中值滤波，在逐渐增大尺寸。被称为自适应策略。

![algo](http://media.innohub.top/190524-algo.png)

![mid](http://media.innohub.top/190524-mid.PNG)



### 2.5 周期降噪

//todo