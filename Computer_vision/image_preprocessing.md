# 图像预处理

[TOC]

## 1. 基本的图像处理

+ 在空间坐标的处理
  + 几何变换
+ 像素处理
  + 代数处理（对于每一个像素点的操作）

## 2. 代数操作

不改变图像的几何性质，对于图像的**灰度级**进行加减乘除等算术运算，逐像素的操作。

### 2.1 生成图像叠加效果（Alpha Blending）

+ 对于两个图像的**合成**可以使用加法进行叠加操作：
![g(x, y) = \alpha*f(x, y) + \beta*f(x, y) \\ \alpha + \beta = 1](http://latex.codecogs.com/gif.latex?g%28x%2C%20y%29%20%3D%20%5Calpha*f%28x%2C%20y%29%20&plus;%20%5Cbeta*f%28x%2C%20y%29%20%5C%5C%20%5Calpha%20&plus;%20%5Cbeta%20%3D%201)
  也可以用于任何两张图像的衔接。
  
+ 使用乘法，可以进行特定区域的提取，可以将除了待提取区域的位置的像素全部置为0
  ![C = \alpha*F + (1-\alpha)B](http://latex.codecogs.com/gif.latex?C%20%3D%20%5Calpha*F%20&plus;%20%281-%5Calpha%29B)
  

### 2.2 获取图像的前景

可以给定一个当前图像*I*,以及对应的背景图像*B*,然后计算两个图像之间的差异：

![Diff(x,y) = ||I(x, y) \; - \; B(x,y)||^2](http://latex.codecogs.com/gif.latex?Diff%28x%2Cy%29%20%3D%20%7C%7CI%28x%2C%20y%29%20%5C%3B%20-%20%5C%3B%20B%28x%2Cy%29%7C%7C%5E2)

如果该差异值大于某一个特定值*T*，就可以视为前景图像.**在背景保持不变的环境下，对于运动目标的检测：可以使用减法操作完成，在差图像不为零的位置表明出现过运动目标。**

## 3.逻辑操作

+ AND: 等价于乘法
+ OR: 等价于乘法
+ XOR: 异或操作
+ NOT： 非

## 4. 几何操作

通过像素位置的变换，运算后直接确定该像素的灰度值的运算，集合运算可以改变图像中的物体之间的位置关系。几何变换需要两个独立的算法：

+ 空间变换：确定变换后的位置
+ 灰度级插值：对于变换后的非整数的位置，以及由于放大、平移、旋转造成的图像中的位缺需要进行插值。

### 4.1 空间变换

要求变换后保持图像中曲线性特征的连续性以及各物体间的连通性。

![g(x,y) = f(\hat{x}, \hat{y}) = f[a(x, y), b(a, y)]](http://latex.codecogs.com/gif.latex?g%28x%2Cy%29%20%3D%20f%28%5Chat%7Bx%7D%2C%20%5Chat%7By%7D%29%20%3D%20f%5Ba%28x%2C%20y%29%2C%20b%28a%2C%20y%29%5D)


### 4.2 灰度级插值

再输入图像中，灰度值仅仅在整数位置（x,y）被定义，但是在上式中的*g(x, y)*的像素值一般由处于非整数位置的像素点确定。所以需要使用取样的方法，确定目标点的灰度值。

+ 最近邻： x = int(x+ 0.5), y = int(y + 0.5)

+ 双线性插值: 假设每一个像素间的颜色变化是线性的
  
> 2次水平，1次垂直
  >
  > 2次垂直， 1此水平

  

+ 双三次插值：假设每一个像素间的颜色分布为三次函数分布，需要确定一个一元三次方程的四个参数，只需要四个点即可

### 4.3 基本几何操作

+ 平移:

  像素点平移向量(x0, y0), 基本变换的公式:

  > a(x, y) = x + x0
  >
  > b(x, y) = x + y0

  

+ 放缩：不是一一映射，为了避免“方块效应”，需要插值操作

  >a(x, y) = s1 * x
  >
  >b(x, y) = s2 * y

+ 旋转:

  绕原点逆时针旋转![\theta](http://latex.codecogs.com/gif.latex?%5Ctheta)角，可以使用以下公式得到变换后的位置坐标：

  > 
  > ![a(x, y) = xcos(\theta) - ysin(\theta) \\
  b(x, y) = xsin(\theta) + ycos(\theta)
  ](http://latex.codecogs.com/gif.latex?a%28x%2C%20y%29%20%3D%20xcos%28%5Ctheta%29%20-%20ysin%28%5Ctheta%29%20%5C%3B%20b%28x%2C%20y%29%20%3D%20xsin%28%5Ctheta%29%20&plus;%20ycos%28%5Ctheta%29)

  

+ 仿射变换

+ Homograph(单应性矩阵)

**向左平移，逆时针旋转为正！**

复杂变换可以使用变换矩阵的形式给出。

## 5. 灰度映射

灰度映射是基于图像像素的点操作，关键在于更具增强的需要设计映射函数。

空间域上的图像增强:

+ 点操作
  + 灰度值变换
  + 直方图操作
+ 空间处理
  + 全局算术操作
  + 基于局部的空间滤波
    + 平滑
    + 锐化

### 5.1 灰度值变换

#### 强度调整

基本的变换函数:

1. 线性变换

2. 对数变换，*s = c×log(1+r)* 其c为常数，r非负。

   ![](http://media.innohub.top/190517-bm.png)

   

3. 幂律变换，![s = c × r^\gamma](http://latex.codecogs.com/gif.latex?s%20%3D%20c%20%D7%20r%5E%5Cgamma), 即Gamma纠正，其中![x, \gamma](http://latex.codecogs.com/gif.latex?x,\gamma)均为正实数

   ![不同的gamma纠正](http://media.innohub.top/190517-gam1.png)

   ![$\gamma$](http://latex.codecogs.com/gif.latex?\gamma)大于1，压缩暗部，增强亮部（**整体亮度减低**）；小于1，压缩亮部（**整体亮度提高**），增强暗部。用以改变对比度。灰度值变换函数的基本要求：连续的增函数，保证变换后从白到黑的顺序不变。

   

#### 对比度调整

二值化方法:

1. isodata 法

   选定一个初始阈值T,将图像分为两部分R1, R2。计算R1, R2 的灰度均值u1, u2；使用(u1+u2)/2作为新的阈值，直到不再变化。

2. Ostu算法

   **最大类间方差法。** 把图像的灰度级数分为两个部分，使得两个部分类间差异最大，类内差异最小。通过计算方差确定一个合适的灰度级别进行划分。

#### 突出特定的灰度级

二值化图像或者仅提亮感兴趣的部分。

#### 位平面切片

### 5.2 直方图均衡化

直方图均衡化的基本思想是**把原始图像的直方图，变换为在整个灰度范围内均匀分布的形式，**这样就增加了像素灰度值的动态范围，从而达到了图像对比度增强的效果。类似于灰度映射，需要一个变换函数，即**增强函数**，该函数的基本要求有两个:

+ 定义域为[0, L-1]的单值单增函数
+ 映射后的范围依旧为[0, L-1]

![$$
s_k = T(r_k) = (L-1)\sum_{j=0}^{k}\frac{n_j}{n} = (L-1)\sum p(r_j)
$$](http://latex.codecogs.com/gif.latex?s_k%20%3D%20T%28r_k%29%20%3D%20%28L-1%29%5Csum_%7Bj%3D0%7D%5E%7Bk%7D%5Cfrac%7Bn_j%7D%7Bn%7D%20%3D%20%28L-1%29%5Csum%20p%28r_j%29)


基本步骤:

1. 归一化，将所有的离散灰度等级进行归一化
2. 每一个累加归一化后的值乘以（L-1）,L为灰度等级，取最近的灰度级作为结果
3. 根据原始的灰度等级以及2中的结果进行映射，得到最终的灰度直方图

### 5.3 直方图规定化

直方图均衡化的优点在于可以自动的增强整个图像的对比度，计算过程没有用户调整的参数。所以具体的增强效果也就无法控制。实际使用中为了有选择的增强某个灰度范围的对比度，或者使图像的灰度等级满足某一个特定的要求。可以使用直方图规定化的方法。

#### 基本步骤:

+ 对原始直方图进行灰度均衡化
+ 规定需要的直方图，并计算可以使得规定直方图均衡化的变换。
+ 将第一步得到的变换翻转过来，也就是将原始直方图对应映射到规定直方图。

#### 主要有两种方式进行映射:

+ 单映射规则（**Single mapping law**）简单，一种有偏的映射规则，有时有较大的取整误差
+ 组映射规则（**Group mapping law**）统计无偏，误差较小

单映射规则要求*原始的累积直方图*的每一项依次向*规定的累计直方图*进行映射，每次选择最接近的数值，**选择最短的直线。**

组映射规则是将规定的累积直方图的每一项，向原始累积直方图进行映射，每次选择最短的直线。





