# Algorithm

[TOC]



## 1. 矩阵分解

### 1.1 隐语义模型(LFM)

Latent Factor Model.通过隐含特征联系用户兴趣以及物品,对于某一个用户首先找到其兴趣分类,然后从兴趣分类中选择他可能喜欢的物品.通过将用户的评分矩阵映射到一个低维的隐语义空间中,挖掘用户的隐因子以及item的隐因子.

隐因子模型即矩阵分解模型:

**SVD:**
$$
R = u * \sum * V^T \\
m*n \; m*r \; r*r \; r*n
$$
传统的SVG要求矩阵稠密，需要对于评分矩阵进行补全，对于高维稀疏的评分矩阵难以应用。

#### FunkSVD

将一个`mxn`的矩阵分解为两个矩阵`P`, `Q`,分别表示用户以及item的隐因子矩阵。

- $p_{ir}$: 表示用户i与隐因子r的相关程度
- $Q_{r, j}$: 表示物品j与隐因子r的相关程度

构建一个损失函数，利用机器学习的方法进行训练:
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1} ^{n} (p_i - r_i)^2}
$$
对于得到的两个低维向量，定义损失函数:
$$
\hat{r} = pq \\
loss(p, q) = \sum _{(u, i) \in Train} (r_{ui} \; - \sum_{f = 1} ^{F} p_{uf} \; q_{fi}) ^2
$$

#### Regularized MF

在基本的矩阵分解的损失函数上加入了正则化，降低过拟合的风险：
$$
loss(p, q) = \sum _{(u, i) \in Train} (r_{ui} \; - \sum_{f = 1} ^{F} p_{uf} \; q_{fi}) ^2 + 
\lambda(||p_u||^2 + ||q_i||^2)
$$
超参数；

- 隐因子的隐含维度F
- 梯度下降的学习率
- 正则化的惩罚因子 $\lambda$
- 正负样本的比例r

#### BiasSVD

BiasSVD 假设评分系统包括三部分的偏置因素：

- 一些与用户物品无关的评分因素
- 用户偏置项：用户有一些和物品无关的评分因素
- 物品偏置项： 物品有一些和用户无关的偏置因素

最终一个用户的评分由4部分组成，原有的矩阵分解结果与系统平均分、用户偏置项、物品偏置项的和
$$
\hat{r_{ui}} = p_u q_i + \mu_u + b_u + b_i
$$
损失函数:
$$
loss(p, q) = \sum _{(u, i) \in Train} (r_{ui} -\mu_u - b_u - b_i\; - \sum_{f = 1} ^{F} p_{uf} \; q_{fi}) ^2 + 
\lambda(||p_u||^2 + ||q_i||^2)
$$

#### SVD++

SVD++ 在BiasSVD方法上进行了改进，考虑到了用户的隐式反馈。充分利用了评分矩阵中的缺失值:

考虑到了邻域的影响:

ItemCF:

+ $N(u)$: 用户u的物品偏好集合
+ S(j, K): 与物品j最相近的K个物品的而集合
+ $w_{ij}$: 物品i，j的相似度
+ $r_{ui}$: 用户u对于物品i的评分

那么基于物品的方法中，用户的评分:
$$
p_{uj} = \sum_{i \in N(u) \cap S(j, K)}w_{ij}r_{ui}
$$
为了求出$w_{ij}$,可以使用一下损失函数:
$$
loss(w) = \sum_{(u,i) \in Train} (r_{ui} - \sum_{j \in N(u)}w_{ij}r_{uj})^2 + \lambda w_{ij}^2 
$$
由于w是一个稠密矩阵，存储是需要较大的空间，所以需要对于w也进行分解：
$$
\hat{r_{ui}} = \frac{1}{\sqrt{|N(u)|}}\sum_{j \in N(u)}w_{ij} \\
 = \frac{1}{\sqrt{|N(u)|}}\sum_{j \in N(u)} x_i^Ty_i = \frac{1}{\sqrt{|N(u)|}} x_i^T \sum_{j \in N(u)} y_i
$$
最终的SVD++：
$$
\hat{r_{ui}} =  \mu_u + b_u + b_i + p_u q_i + \frac{1}{\sqrt{|N(u)|}} x_i^T \sum_{j \in N(u)} y_i \\
\hat{r_{ui}} =  \mu_u + b_u + b_i + p_u( q_i + \frac{1}{\sqrt{|N(u)|}}\sum_{j \in N(u)} y_i)
$$
**用户兴趣：显式兴趣 + 偏见 + 隐式反馈**

