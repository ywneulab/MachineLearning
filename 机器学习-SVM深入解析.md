---
title: SVM深入解析
sitemap: true
categories: 机器学习
date: 2018-10-30 10:48:31
tags:
- 机器学习
- SVM
- 核函数
- SMO
---

# 前言

起初让我最头疼的是拉格朗日对偶和SMO，后来逐渐明白拉格朗日对偶的重要作用是将w的计算提前并消除w，使得优化函数变为拉格朗日乘子的单一参数优化问题。而SMO里面迭代公式的推导也着实让我花费了不少时间。

对比这么复杂的推导过程，SVM的思想确实那么简单。它不再像logistic回归一样企图去拟合样本点（中间加了一层sigmoid函数变换），而是就在样本中去找分隔线，为了评判哪条分界线更好，引入了几何间隔最大化的目标。

之后所有的推导都是去解决目标函数的最优化上了。在解决最优化的过程中，发现了w可以由特征向量内积来表示，进而发现了核函数，仅需要调整核函数就可以将特征进行低维到高维的变换，在低维上进行计算，实质结果表现在高维上。由于并不是所有的样本都可分，为了保证SVM的通用性，进行了软间隔的处理，导致的结果就是将优化问题变得更加复杂，然而惊奇的是松弛变量没有出现在最后的目标函数中。最后的优化求解问题，也被拉格朗日对偶和SMO算法化解，使SVM趋向于完美。

<span id = "简述 SVM 的基本概念和原理">
# 简述 SVM 的基本概念和原理

最简单的 SVM 从线性分类器导出, 根据最大化样本点分类间隔的目标, 我们可以得到线性可分问题的 SVM 目标函数. 然后可以利用拉格朗日乘子法得到其对偶问题, 并根据 KKT 条件和 SMO 算法就可以高效的求出超平面的解. 但是实际任务中, 原始样本空间内也许并不存在一个能正确划分两类样本的超平面. 因此, 我们需要利用核函数将样本从原始空间映射到一个更高为的特征空间, 使得样本在这个特征空间内线性可分. 核函数的选择对于支持向量机的性能至关重要. 但是现实任务中往往很难确定合适的核函数使得训练样本在特征空间内线性可分, 因此, 我们引入了 "软间隔" 的概念, 也就是松弛变量和惩罚因子, 其基本思想就是, 允许支持向量机在一些样本上出错, 并对违反约束条件的训练样本进行惩罚. 所以, 最终的优化目标就是在最大化间隔的同时, 使得不满足约束的样本尽可能地少.

<span id = "SVM 推导过程">
# SVM 推导过程

# 1 间隔与支持向量

给定训练样本集(二分类问题):
$$D = {(\vec x_1, y_1), (\vec x_2,y_2),..,(\vec x_m,y_m)}$$
$$y_i \in \{-1, +1\}$$
$$\vec x_i =(x^{(1)}_i;x^{(2)}_i;...;x^{(d)}_i )$$
注意,这里用的是分号, 表示这是一个列向量. SVM做的事情就是试图把一根 "木棍" 放在最佳位置, 好让 "木棍" 的两边都有尽可能大的 "间隔".

这个 "木棍" 就叫做 "划分超平面", 可以用下面的线性方程来描述:

$$\vec w^T\vec x + b = 0$$

其中 $\vec w =(w^(1); w^(2);...;  w^(d))$ 为 $d$ 维法向量(注意,这里用的是分号, 表示这是一个列向量), **决定了超平面的方向**,  $\vec x$ 为 "木棍" 上的点的坐标($d$ 维列向量), $b$ 为位移项, **决定了超平面与原点之间的距离**.

根据点到 "直线" 的距离公式,我们可以得到样本空间中任意点 $\vec x$ 到超平面 $(\vec w,b)$ 的距离为:

$$r = \frac{|\vec w^T\vec x+b|}{\|\vec w \|}$$

$\|\|\vec w \|\| = \sqrt{w_1^2 + w_2^2 + ... + w_d^2}$ 为向量长度(也即向量的L2范式)

**首先假设** 当前的超平面可以将所有的训练样本正确分类, 那么就有如下式子:

$$\begin{cases} \vec w^T\vec x_i + b \geq 0, & y_i = +1 \\ \vec w^T\vec x_i + b < 0, & y_i = -1 \end{cases}$$

上式可以统一写成如下的约束不等式:

$$y_i (\vec w^T\vec x_i + b) \geq 0$$

上面的式子其实是冗余的, 因为假设样本点不在超平面上, 所以不可能出现等于0的情况, 又因为超平面方程两边都乘一个不等于0的数,还是 **同一个超平面**, 因此为了简化问题的表述, 我们对 $\vec w$ 和 $b$ 加上如下约束(这里的1没有什么特别的含义, 可以是任意的常数, 因为这里的点 $\vec x_i$ 不是超平面上的点, 所以所得值不为0):

$$\min_i|\vec w^T\vec x_i +b| = 1$$

即离超平面最近的正, 负样本距离超平面的距离为: $\frac{1}{\|\|\vec w\|\|}$ , 我们将这些距离超平面最近的几个训练样本点为定义 "支持向量", 那么, 两个异类支持向量到超平面的距离之和就为 $\gamma = \frac{2}{\|\|\vec w\|\|}$ , 我们将这称为"间隔".

同时, 根据此约束, 我们可以消除超分类平面约束的冗余, 得到新的超分类平面约束如下:

$$y_i(\vec w^T\vec x_i + b) \geq 1$$

SVM的目的就是找到具有 "最大间隔" 的划分超平面, 也就是要找到满足约束 $y_i(\vec w^T\vec x_i + b) \geq 1$ 中的参数 $\vec w, b$ , 使得其具有最大的间隔 $\gamma$ , 也就是:

$$\arg\max_{\vec w,b}\frac{2}{\|\vec w\|}$$
$$s.t. y_i(\vec w^T \vec x_i +b) \geq 1, i=1,...,m$$

显然, 为了最大化间隔 $\gamma$ , 我们仅需要最大化 $\|\vec w\|^{-1}$ , 这就等于最小化 $\|\vec w\|^2$, 于是上式等价为:

$$\arg\min_{\vec w,b} \frac{1}{2}\|\vec w\|^2 = \arg\min_{\vec w,b} \frac{1}{2}\vec w^T\vec w  \tag 1$$
$$s.t. y_i(\vec w^T \vec x_i +b) \geq 1, i=1,...,m$$

下图即为SVM示意图, 注意,图中的1可以被任意常数替换(只要前面乘上对应的系数即可, =0说明在超分类平面上, !=0说明在两侧)

![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fvrusa33dkj20hx0dnmxr.jpg)

以上就是线性可分时的SVM基本型(现实中大多数问题是线性不可分的, 所以线性可分的SVM没有太多实用价值)

# 2 对偶问题求解 $\vec w$ 和 $b$

## 问题说明

对偶问题(dual problem):在求出一个问题解的同时, 也给出了另一个问题的解

我们希望通过求解式(1)来得到具有最大间隔的划分超平面的模型参数,由于该式是一个凸二次规划问题(目标函数是变量的二次函数, 约束条件是变量的线性不等式). 因此,对该式使用拉格朗日乘子法得到其 "对偶问题".

对于式(1)的 **每个样本点** 约束添加拉格朗日乘子 $\alpha_i \geq 0$, 则该问题的拉格朗日函数为:

$$L(\vec w,b,\alpha) = \frac{1}{2}\|\vec w\|^2 +\sum_{i=1}^{m}\alpha_i (1-y_i(\vec w^T \vec x_i +b))\tag 2$$

其中, $\vec \alpha = (\alpha_1, \alpha_2,...,\alpha_m)$ ,每一个 $\alpha_i$ 均为标量 .接着令 $L(\vec w,b,\vec \alpha)$ 对 $\vec w$ 和 $b$ 求偏导, 并令其为0, 可得:

$$\frac{\partial L(\vec w,b,\vec \alpha)}{\partial \vec w} = \vec w - \sum_{i=1}^{m} \alpha_i y_i \vec x_i = 0 \tag 3$$

$$\frac{\partial L(\vec w,b,\vec \alpha)}{\partial b} = -\sum_{i=1}^{m}\alpha_i y_i = 0 \tag 4$$

将(3)和(4)代入(2)式中, 消去 $\vec w$ 和 $b$ ( 注意, 这里 $\sum_{i=1}^{m}\alpha_i y_i = 0$, 但是不代表 $\alpha_i y_i = 0$ ), 可得:

$$L(\vec w, b, \vec \alpha) = \frac{1}{2}\bigg( \sum_{i=1}^{m}\alpha_i y_i \vec x_i \bigg)^2 + \sum_{i=1}^{m} \alpha_i - \sum_{i=1}^{m}\alpha_i y_i \Big( \sum_{j=1}^{m} \alpha_j y_j \vec x_j \Big)^T \vec x_i - \sum_{i=1}^{m} \alpha_i y_i b$$

$$= \sum_{i=1}^{m}\alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i y_j \alpha_i y_j \vec x_i^T \vec x_j $$

这里 $\vec x_i,\vec x_j$ 位置可互换, 为了好看,我将 $\vec x_i$ 写在了前面. 到此, 我们就得到了式(2)的对偶问题:

$$\arg\max_{\vec \alpha}  \bigg( \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i  y_j \vec x_i^T \vec x_j \bigg) \tag 5$$
$$s.t. \sum_{i=1}^{m} \alpha_i y_i = 0, 其中 \alpha_i \geq 0$$

为了满足原始问题(1) 和对偶问题(5)之间的充分必要条件, 上述推导过程还需要满足KKT(Karush-Kuhn-Tucker)条件(其中前两条已经在上述推导过程中满足) , 即要求:

$$\begin{cases} \alpha_i \geq 0 ; \\ y_i f(\vec x_i) - 1 \geq 0 ; \\ \alpha_i (y_i f(\vec x_i) - 1 ) = 0. \end{cases}$$

当我们解出上式得到 $\vec \alpha$ 后, 就可以通过求得 $\vec w$ 和 $b$ 的值,  进而可得到划分超平面对应的模型:

$$ f(\vec x) = \vec w^T \vec x +b = \sum_{i=1}^{m} \alpha_i y_i \vec x_i^T \vec x +b$$

根据 KKT 条件我们可以轻易得出, 对任意的训练样本 $(\vec x_i , y_i)$ , 总有 $\alpha_i = 0$ 或 $y_i f(\vec x_i) = 1$ . 若 $\alpha_i = 0$ , 则该项对应的样本不会出现在求和项中 ; **若 $\alpha_i > 0$ , 则必有 $y_i f(\vec x_i) = 1$ , 这说明该样本点出现在最大间隔边界上, 是一个支持向量. 这显示出支持向量机的一个重要性质: 训练完成后, 大部分的训练样本都不需要保留(这些样本对应的系数 $\alpha_i = 0$ ), 最终模型仅与支持向量有关.**

## 使用SMO算法求对偶问题的解

从(5)式可以看出, 这仍是一个二次规划问题, 可以使用通用的二次规划法来求解, 但是, 该问题的规模正比于训练样本数量, 在实际任务中使用通用解法会造成很大的开销, 因此, 需要使用更高效的算法---SMO(Sequential Minimal Optimization, 序列最小算法)

SMO的基本思路: 先固定 $\alpha_i$ 之外的所有参数, 然后求 $\alpha_i$ 上的极值.  但是这里由于 $\alpha_i$ 之间不是互相独立的, 需要满足约束 $\sum_{i=1}^{m} \alpha_i y_i = 0$ , 即一个分量改变, 其他的也要随之改变,因此每次在优化变量中选取两个分量 $\alpha_i$ 和 $\alpha_j$ ,并将其他参数固定, 然后在参数初始化后, 不断执行如下两个步骤直至收敛:
- 选取一对需要更新的变量 $\alpha_i$ 和 $\alpha_j$
- 固定 $\alpha_i$ 和 $\alpha_j$ 以外的参数, 求解(5)式更新后的 $\alpha_i$ 和 $\alpha_j$

具体的求解过程如下:

首先假设需要优化的参数是 $\alpha_i$ 和 $\alpha_j$ , 于是我们将剩下的分量 $\sum_{k=1,k \neq i,j}^{m} \alpha_k y_k$ 固定, 作为常数处理, 可得下式:

$$\alpha_i y_i + \alpha_j y_j = -\sum_{k\neq i,j}^{m} \alpha_k y_k = C$$

对上式两边同乘以 $y_j$ ,由于 $y_j \times y_j = 1$ 可得:

$$\alpha_j = C y_j - \alpha_i y_i y_j = y_j(C - \alpha_i y_i)$$


将上式代入(5)式, 消去变量 $\alpha_j$ , 得到一个关于 $\alpha_i$ 的单变量二次规划问题, 所有的常数项用 $C$ 表示, (5)式被转换成如下,:

$$F(\alpha_i) = \alpha_i + \Big( y_j(C - \alpha_i y_i) \Big) - \frac{1}{2}\alpha_i \alpha_i y_iy_i\vec x^{(i)T}\vec x_i - \frac{1}{2}\Big( y_j(C - \alpha_i y_i) \Big)^2y_jy_j\vec x^{(j)T}\vec x_j $$
$$-  \alpha_i \Big( y_j(C - \alpha_i y_i) \Big) y_iy_j\vec x^{(i)T} \vec x_j$$
$$- \alpha_iy_i\sum_{k=1,k\neq i,j}^{m}\alpha^{(k)}y^{(k)}\vec x^{(i)T} \vec x^{(k)} - \Big( y_j(C - \alpha_i y_i) \Big) y_j\sum_{k=1,k\neq i,j}^{m}\alpha^{(k)}y^{(k)}\vec x^{(j)T}\vec x^{(k)}$$
$$= \alpha_i + \Big( y_j(C - \alpha_i y_i) \Big) - \frac{1}{2}(\alpha_i)^2\vec x^{(i)T}\vec x_i - \frac{1}{2} \big( C - \alpha_iy_i \big)^2 \vec x^{(j)T}\vec x_j - \alpha_i \Big( (C - \alpha_i y_i) \Big) y_i\vec x^{(i)T} \vec x_j - \alpha_iy_iv_i - \big(C- \alpha_iy_i \big)v_j + C$$
$$= \alpha_i + \Big( y_j(C - \alpha_i y_i) \Big) - \frac{1}{2}(\alpha_i)^2K_{i,i} - \frac{1}{2} \big( C - \alpha_iy_i \big)^2 K_{j,j} - \alpha_i \Big( (C - \alpha_i y_i) \Big) y_iK_{i,j} - \alpha_iy_iv_i - \big(C- \alpha_iy_i \big)v_j + C$$
上式为了简便, 将 $\vec x^{(i)T}\vec x_j$ 简记为 $K_{i,j}$ (后文会用K代表核函数, 这里姑且认为此时的核函数 $K$ 为恒等映射),将上式对 $\alpha_i$ 求导, 并令其等于0, 可得:

$$\frac{\partial F(\alpha_i)}{\partial \alpha_i} = 1 - y_iy_j - \alpha_iK_{i,i} + y_i(C-\alpha_i y_i)K_{j,j} - \Big( C-\alpha_iy_i - \alpha_i y_i \Big)y_iK_{i,j} - y_iv_i + y_iv_j$$

$$=  1-y_iy_j -\alpha_i \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big) + Cy_iK_{j,j} - Cy_iK_{i,j} - y_i\big(v_i -v_j \big)  = 0$$

下面对上式进行变形, 使得可以用 $\alpha_i^{old}$ 来更新 $\alpha_i^{new}$ .

因为SVM对数据点的预测值为: $f(\vec x) = \sum_{i=1}^{m}\alpha_i y_i K(\vec x_i,\vec x) + b$, 则 $v_i$ 以及 $v_j$ 的值可以表示成:

$$ v_i = \sum_{k=1,k\neq i,j}^{m} \alpha^{(k)} y^{(k)} K_{i,k} = f(x_i) - \alpha_i y_i K_{i,i} - \alpha_j y_j K_{i,j} + b$$

$$ v_j = \sum_{k=1,k\neq i,j}^{m} \alpha^{(k)} y^{(k)} K_{j,k} = f(x_j) - \alpha_j y_j K_{j,j} - \alpha_i y_i K_{j,i} + b$$

将 $\alpha_j = y_j(C - \alpha_i y_i)$ 带到上式, 可得到 $v_i - v_j$ 的表达式为:


$$v_i - v_j = f(x_i) - f(x_j) - \alpha_i y_i K_{i,i} + \Big( y_j(C - \alpha_i y_i) \Big) y_j K_{j,j} - \Big( y_j(C - \alpha_i y_i) \Big) y_jK_{i,j} + \alpha_iy_iK_{j,i}$$

$$= f(x_i) - f(x_j) - \alpha_iy_iK_{i,i} + CK_{j,j} - \alpha_iy_iK_{j,j} - CK_{i,j} + 2\alpha_iy_iK_{i,j}$$

$$ = f(x_i) - f(x_j) - \alpha_iy_i \Big( K_{i,i} + K_{j,j} -2K_{i,j} \Big)+ CK_{j,j} - CK_{i,j}$$

**注意 $v_i - v_j$ 中 $\alpha_i$ 是更新前初始化的值, 我们将其记作 $\alpha_i^{old}$ ,以便与我们期望获得的更新后的分量 $\alpha_i^{new}$ 相区分** , 将 $v_i - v_j$ 的表达式代入 $\frac{\partial F(\alpha_i)}{\partial \alpha_i^{new}}$ 中 , 可得到:

$$\frac{\partial F(\alpha_i^{new})}{\partial \alpha_i^{new}} = 1-y_iy_j -\alpha_i^{new} \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big) + Cy_iK_{j,j} - Cy_iK_{i,j}$$
$$- y_i\bigg (f(x_i) - f(x_j) - \alpha_i^{old}y_i \Big( K_{i,i} + K_{j,j} -2K_{i,j} \Big)+ CK_{j,j} - CK_{i,j} \bigg)$$
$$= \big( y_i \big)^2 -y_iy_j - y_if(x_i) + y_if(x_j) - \alpha_i^{new} \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big) + \alpha_i^{old} \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big)$$
$$=  f(x_j) - y_j - \big( f(x_i) -y_i \big) - \alpha_i^{new} \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big) + \alpha_i^{old} \Big( K_{i,i} + K_{j,j} - 2K_{i,j}\Big)$$

我们记 $E_i$ 为SVM预测值与真实值的误差: $E_i = f(x_i) - y_i$ . 并令 $\eta = K_{i,i} + K_{j,j} - 2K_{i,j}$ , 则最终的一阶导数表达式可以简化为:

$$\frac{\partial F(\alpha_i^{new})}{\partial \alpha_i^{new}} = -\eta \alpha_i^{new} + \eta \alpha_i^{old} + y_i\big(E_j - E_i \big) = 0$$

由此, 我们可以根据当前的参数值, 直接得到更新后的参数值:

$$ \alpha_i^{new} = \alpha_i^{old} + \frac{y_i\big(E_j - E_i \big)}{\eta} => \alpha_i^{new, unclipped} \tag 6$$

这里注意, (6)式的推导过程并未考虑下面的约束, 因此, 我们暂且将(6)式中的 $\alpha_i^{new}$ 记作 $\alpha_i^{new, unclipped}$, 然后考虑如下约束:

$$\alpha_i y_i + \alpha_j y_j = -\sum_{k=1,k\neq i,j}^{m} \alpha^{(k)} y^{(k)} = C$$

$$0 \leq \alpha_i , \alpha_j \leq C$$

我们分别以 $\alpha_i, \alpha_j$ 为坐标轴, 于是上述约束可以看作是一个方形约束(Bosk constraint), 在二维平面中我们可以看到这是个限制在方形区域中的直线, 如下图所示, 直线在方形区域内滑动(对应不同的截距), 同时 $\alpha_i^{new}$ 的上下边界也在改变:

![](https://wx4.sinaimg.cn/mw690/d7b90c85ly1fvsu4x9m5hj20ci0a8wel.jpg) ![](https://wx3.sinaimg.cn/mw690/d7b90c85ly1fvsu4xaa2nj20ca0ae3yn.jpg)

当 $y_i \neq y_j$ 时(如左图), 限制条件可以写成 $\alpha_i - \alpha_j = \xi$ ,根据 $\xi$ 的正负可以得到不同的上下界,   因此 $\alpha_i^{new}$ 的上下界可以统一表示成:
- 下界: $L = \max(0, \alpha_i^{old} - \alpha_j^{old})$
- 上界: $H = \min(C, C + \alpha_i^{old} - \alpha_j^{old})$

当 $y_i = y_j$ 时(如右图), 限制条件可以写成 $\alpha_i + \alpha_j = \xi$ , 于是 $\alpha_i^{new}$ 的上下界为:
- 下界: $L = \max(0,\alpha_i^{old} + \alpha_j^{old} - C)$
- 上界: $H = \min(C, \alpha_i^{old} + \alpha_j^{old})$


根据得到的上下界, 我们可以得到"修剪"后的 $\alpha_i^{new,clipped}$ :

$$\alpha_i^{new,clipped} = \begin{cases} H & \alpha_i^{new,unclipped} > H \\ \alpha_i^{new,unclipped} & L \leq \alpha_i^{new,unclipped} \leq H \\ L & \alpha_i^{new,unclipped} < L \end{cases} \tag 7$$

得到了 $\alpha_i^{new,clipped}$ 以后, 便可以根据 $\alpha_i^{old} y_i + \alpha_j^{old} y_j= \alpha_i^{new}y_i + \alpha_j^{new}y_j$ 得到 $\alpha_j^{new}$ :

$$\alpha_j^{new,clipped} = \alpha_j^{old} + y_iy_j\big( \alpha_i^{old} - \alpha_i^{new,clipped}  \big) \tag 8$$

通过(7)(8)式, 我们便可以高效的计算出更新后的 $\alpha_i$ 和 $\alpha_j$ .

当更新了一对 $\alpha_i$ 和 $\alpha_j$ 之后, 我们需要计算偏移项 $b$ 注意到, 对于任意支持向量 $(\vec x^{(s)} , y^{(s)})$ , 都有 $y^{(s)} f(x^{(s)}) = 1$ , 即:

$$y^{(s)} \Big( \sum_{i \in S} \alpha_i y_i \vec x^{(i)T} \vec x^{(s)} + b\Big) = 1$$

式中 $S$ 为所有支持向量的下标集. 理论上, 可以选取任意支持向量来获得 $b$ , 但现实中我们采取更加鲁棒的做法: 使用所有支持向量求解的平均值(式中所有量均已知, $\vec \alpha$ 使用的是支持向量对应的系数):

$$b = \frac{1}{|S|} \sum_{s\in S} \bigg( \frac{1}{y^{(s)}} - \sum_{i \in S} \alpha_i y_i\vec x^{(i)T} \vec x^{(s)} \bigg)$$


还有另一种更新 $b$ 的方式是, 只使用当前更新的变量 $\alpha_i^{new}$ 和 $\alpha_j^{new}$ 来对 $b$ 进行更新,如此一来, 为了满足KKT条件, 就有以下几种情况:
- 如果 $\alpha_i^{new}$ 在界内(即此时 $0 < \alpha_i^{new} < C$ , 当前对应样本为支持向量), 则 $b = b_i^{new}$
- 如果 $\alpha_j^{new}$ 在界内(即此时 $0 < \alpha_j^{new} < C$ , 当前对应样本为支持向量), 则 $b = b_j^{new}$
- 如果 $\alpha_i^{new}$ 和 $\alpha_j^{new}$ 都在界上,且 $L \neq H$时, 则 $b_i^{new}$ 和 $b_j^{new}$ 之间的所有的值都符合KKT条件, SMO一般选择终点作为新的偏移量: $b_{new} = \frac{b_i^{new} + b_j^{new}}{2}$

以上讨论中, $b_i^{new}$ 的推导过程为, 当 $\alpha_i^{new}$ 在界内时, 对应的样本为支持向量 (根据KKT条件得出) , 此时 $y_i(\vec w^T \vec x_i +b) = 1$ , 两边同时乘上 $y_i$ ,得到 $\sum_{k=1}^{m}\alpha^{(k)}y^{(k)}K_{k,i} + b = y_i$, 将该式展开, 得到:

$$b_i^{new} = y_i - \sum_{k=1,k\neq i,j}^{m} \alpha^{(k)} y^{(k)}K_{k,i} - \alpha_i^{new}y_iK_{i,i} - \alpha_j^{new}y_jK_{j,i}$$

其中前两项可以写成:

$$y_i - \sum_{k=1,k\neq i,j}^{m} \alpha^{(k)} y^{(k)}K_{k,i} = -E_i + \alpha_i^{old}y_iK_{i,i} + \alpha_j^{old}y_jK_{j,i} + b_{old}$$

于是有:

$$b_i^{new} = -E_i - \big( \alpha_i^{new} - \alpha_i^{old} \big)y_i  K_{i,i} - \big(\alpha_j^{new} - \alpha_j^{old}   \big)y_jK_{j,i} + b_{old}$$

同理有:
$$b_j^{new} = -E_j - \big( \alpha_j^{new} - \alpha_j^{old} \big)y_j  K_{j,j} - \big(\alpha_i^{new} - \alpha_i^{old}   \big)y_jK_{i,j} + b_{old}$$



## 如何恰当的选取需要更新的变量 $\alpha_i$ 和 $\alpha_j$

采用启发式的规则来选取, 直觉上我们知道, 我们应该首先优化那些违反KKT条件最严重的样本, 因此我们首先首先遍历所有满足约束条件 $0 < \alpha_i < C$ 的样本点, 即位于间隔边界上的支持向量点(直觉上也能发现这些点最有可能分类错误), 检验它们是否满足KKT条件. 如果这些样本都满足KKT条件，则遍历整个训练样本集，判断它们是否满足KKT条件，直到找到一个违反KKT条件的变量 $\alpha_i$ (即使 $\alpha_i$ 位于边界上,也有可能违反KKT条件).

当找到了第一个分量 $\alpha_i$ 后, 接下来寻找第二个分类 $\alpha_j$, 而选取的标准是使得它有足够大的变化, 也就是说使选取的两变量所对应的样本之间的间隔最大, 一种直观的解释是, 这样的两个变量有很大的差别, 与对两个相似的变量进行更新相比(相似说明有可能属于同一类, 更新意义不大), 对它们进行更新会带给目标函数值更大的变化. 第二个乘子的迭代步长正比于 $|E_i - E_j|$ , 因此, 我们希望选择的乘子能够具有最大的 $|E_i - E_j|$. 即当 $E_i$ 为正时选择绝对值最大的赋值 $E_j$ , 反之, 选择正值最大的 $E_i$


# 3 核函数

在之前的讨论中,我们假设 **训练样本** 是线性可分的, 然而在现实任务中, 原始样本空间内也许并不存在一个能正确划分两类样本的超平面, 对于这样的问题, **可将一样本从原始空间映射到一个更高维的特征空间, 使得样本在这个特征空间内线性可分** .

**需要知道, 如果原始空间是有限维, 即属性数有限, 那么一定存在一个高维特征空间使样本可分**

令 $\phi(\vec x)$ 表示将 $\vec x$ 映射后的特征向量, 于是, 在特征空间中划分超平面所对应的模型可表示为:

$$f(\vec x) = \vec w^T \phi(\vec x) + b$$

类似式(1), 有:

$$\arg\min_{\vec w,b} \frac{1}{2} \|w\|^2$$

$$ s.t. y_i\big( \vec w^T \phi (\vec x_i) + b \big), i=1,2,..,m$$

其对偶问题为:

$$\arg\max_{\vec \alpha} = \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \phi(\vec x_i)^T \phi(\vec x_j) \tag 9$$

$$s.t. \sum_{i=1}^{m} \alpha_i y_i = 0, \alpha_i \geq 0 , i = 1,2,...,m$$

求解上式涉及到计算 $\phi(\vec x_i)^T \phi(\vec x_j$ , 这是样本 $\vec x_i$ 与 $\vec x_j$ 映射到特征空间之后的内积, 由于特征空间维数可能很高, 甚至是无穷维, 因此直接计算 $\phi(\vec x_i)^T \phi(\vec x_j$ 是很困难的, 为了避开这个障碍, 可以设想这样一个函数:

$$ K \big(\vec x_i, \vec x_j \big) = \phi(\vec x_i)^T \phi(\vec x_j$$

**即 $x_i$ 与 $x_j$ 在特征空间的内积等于它们在原始样本空间中通过函数 $K(\cdot, \cdot)$ 计算的结果.** (有可能是先内积再函数映射, 也有可能是求范式再函数映射). 于是(9)式可重写为:

$$\arg\max_{\vec \alpha} \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m} \sum_{j=1}^{m}\alpha_i \alpha_j y_i y_j K\big(\vec x_i, \vec x_j \big)$$

$$s.t. \sum_{i=1}^{m} \alpha_i y_i = 0$$

$$ \alpha_i \geq 0, i=1,2,...,m$$

注意, 前面几个小节的推导过程也用了符号 $K$ , 但是就像前面所说的, 前几个小节的 $K$ 是为了方便书写而使用的, 你可以把它看作是一个恒等映射的核函数

当我们解出上式得到 $\vec \alpha$ 后, 就可以得到划分超平面对应的模型(式中 $\vec x$ 为样本点, $f(\vec x)$ 为该样本点的预测结果):

$$ f(\vec x) = \vec w ^T \vec x +b = \sum_{i=1}^{m} \alpha_i y_i K\big(\vec x, \vec x_j
\big) +b$$

**核函数定理:** 令 $\chi$ 为输入空间 $K(\cdot, \cdot)$　是定义在　$\chi \times \chi$ 上的对称函数, 则 $K(\cdot, \cdot)$ 是核函数 当且仅当 对于任意数据  $D =  \\{\vec x^{(1)}, \vec x^{(2)},...,\vec x ^{(m)} \\}$ , 核矩阵 $K$ 总是半正定的

从以上分析可知, 核函数的选择决定了特征空间的好坏, 因此, 一个合适的核函数,就成为了支持向量机的最大变数.

下面是几种常用的核函数:

| 名称 | 表达式 | 参数 |
| --- | --- | --- |
| 线性核   |   |   |
| 高斯核  |   |   |
| 拉普拉斯核   |   |   |
| Sigoid核   |   |   |

此外,还可以通过函数组合得到:
- 若 $K_1$ 和 $K_2$ 都是核函数 ,则对任意的正数 $\gamma_1, \gamma_2$ , 其线性组合 $\gamma_1 K_1 + \gamma_2 K_2$ 也是核函数
- 若 $K_1$ 和 $K_2$ 为核函数, 则函数的直积 $K_1 \otimes K_2 (\vec x , \vec z) = K_1(\vec x, \vec z) K_2(\vec x, \vec z)$
- 若 $K_1$ 是核函数, 则对任意函数 $g(\vec x)$, $K(\vec x, \vec z) = g(\vec x) K_1(\vec x, \vec z) g(\vec z)$ 也是核函数

# 4 软间隔与正则化

在实现任务中, 往往很难确定合适的核函数, 使得训练样本在特征空间中线性可分, 即便是找到了, 也无法断定是否是由于过拟合造成的 , 因此, 我们需要 **允许支持向量机在一些样本上出错** , 以缓解上面的问题.

硬间隔(hard margin)与软间隔(soft margin)的区分:
- 硬间隔: 所有样本都必须分类正确
- 软间隔: 允许某些样本不满足约束(11)式(即,预测结果和真实结果符号相反,分类错误,或预测结果绝对值小于1,相当于越过了支持向量划定的边界)

我们要在最大化间隔的同时, 使得不满足约束的样本应尽可能的少, 于是, 优化目标可写为:

$$ \min_{\vec w,b} \frac{1}{2} \|w\|^2 + C\sum_{i=1}^{m} l_{0/1} \big( y_i (\vec w^T x_i+b) - 1\big) \tag {10}$$

$$ y_i (\vec w^T \vec x_i +b) \geq 1 \tag {11}$$

其中, $C>0$ 为惩罚因子, 是一个常数(注意与前几节推导SVM时的常数区分), $l_{0/1}$ 是 "0/1 损失函数":

$$l_{0/1} (z) = \begin{cases}  1, & \text{if }
 z < 0 ; \\ 0, & \text{otherwise}. \end{cases}$$

当C无穷大时, (10)式就会迫使所有样本均满足约束, 也就是令所有训练样本都分类正确(容易产生过拟合), 当C取有限值时, 则允许有一些样本不满足约束(11)式.



但是, $l_{0/1}$ 非凸, 不连续, 数学性质不好, 因此, 通常使用其他函数来替代, 称为" 替代损失",  下面为三种常用的替代损失:

- hinge损失: $l_{hinge}(z) = max(0,1-z)$
- 指数损失(exponential loss): $l_{exp}(z) = exp(-z)$
- 对率损失(logistic loss): $l_{log}(z) = log(1+ exp(-z))$

假设采用hinge损失损失, 然后可以引入"松弛变量"(slack variables) $\xi_i \geq 0$ ,每一个样本都有一个对应的松弛变量, 用以表征该样本不满足约束(11)的程度 则可将(10)式重写为:

$$\min_{\vec w, b, \xi_i} \frac{1}{2} \|\vec w\|^2 + C \sum_{i=1}^{m} \xi_i \tag {12}$$

$$ s.t. y_i (\vec w^T x_i + b) \geq 1- \xi_i$$

$$\xi_i \geq , i=1,2,...,m.$$

可以看出, 上式是与之前推导相似的二次规划问题,  只不过是约束条件变的宽松了(为了允许一些样本犯错), 因此,同样利用拉格朗日乘子法求解, 首先得到上式的拉格朗日函数:

$$L(\vec w, b, \vec \alpha, \vec \xi, \vec \mu) = \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i + \sum_{i=1}^{m}\alpha_i\big(1- \xi_i - y_i(\vec w^T\vec x_i +b)  \big) - \sum_{i=1}^{m} \mu_i \xi_i$$

其中, $\alpha_i \geq 0, \mu_i \geq 0$ 是拉格朗日乘子, 令 $L(\vec w, b, \vec \alpha, \vec \xi, \vec \mu)$ 对 $\vec w, b, \vec \alpha, \vec \xi$ 求偏导, 并令其为0 , 可得:

$$\vec w =\sum_{i=1}^{m} \alpha_i y_i \vec x_i$$

$$0 = \sum_{i=1}^{m} \alpha_i y_i$$

$$C = \alpha_i + \mu_i$$


 得到(12)式对应的对偶问题如下:

$$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j K_{i,j}$$

$$s.t. \sum_{i=1}^{m} \alpha_i y_i = 0$$

$$0 \leq \alpha_i \leq C , i=1,2,...,m$$

可以看到, 此时, $\alpha_i$ 的约束条件变成了 $0 \leq \alpha_i \leq C$ , 上式的KKT条件要求为:

$$\begin{cases} \alpha_i \geq 0,  \mu_i \geq 0 \\ y_if(\vec x_i) -1 +\xi_i \geq 0,  \\ \alpha_i \big( y_if(\vec x_i) - 1 + \xi_i \big) = 0, \\ \xi_i \geq 0, \mu_i \xi_i = 0 \end{cases}$$

于是, 从KKT条件中我们可以看出, 对任意的训练样本 $(\vec x_i, y_i)$, 总有 $\alpha_i = 0$ 或 $y_i f(\vec x_i) = 1 - \xi_i$.
- 若 $\alpha_i = 0$, 则该样本不会对 $f(\vec x)$ 产生影响.
- 若 $\alpha_i > 0$, 则必有 $y_i f(\vec x_i) = 1 - \xi_i$, 即该样本是支持向量
- 因为 $C = \alpha_i + \mu_i$ , 所以, 若 $\alpha_i < C$ , 则有 $\mu_i > 0$ , 进而有 $\xi_i = 0$, 即该样本在最大间隔边界上(是否也就是支持向量?)
- 若 $\alpha_i = C$ , 则有 $\mu_i = 0$, 此时若 $\xi_i \leq 1$, 则该样本落在最大间隔内部, 若 $\xi_i > 1$, 则该样本被错误分类.

以上讨论, 我们可以看出, 最终的模型依然只与支持向量有关, 保持了稀疏性(hinge损失有一块平坦的零区域,这使得SVM的解具有稀疏性)

以上是对使用hinge损失时讨论的情况, 还可以将其替换成别的损失函数以得到其他学习模型, 这些模型的性质与所用的替代函数直接相关, 但它们具有一个共性: 优化目标中的第一项用来描述划分超平面的"间隔"大小, 另一项用来表示训练集上的误差, 可写为更一般的形式:

$$ \min_{f} \Omega(f) + C\sum_{i=1}^{m} l(f(\vec x_i) , y_i)$$

其中, $\Omega(f)$ 称为"结构风险"(structural risk), 用于描述模型 $f$ 自身的性质; 第二项 $C\sum_{i=1}^{m} l(f(\vec x_i)$ 称为"经验风险"(empirical risk), 用于描述模型与训练数据的契合程度. $C$ 用于对二者进行折衷.

从预测误差的角度来看, 第二项相当于模型误差, 第一项相当于正则化项, 表述了模型本身的性质, 一方面, 这为引入领域知识和用户意图提供了途径, 另一方面, 该信息有助于消减假设空间, 降低过拟合风险

<span id = "SVM 如何解决线性不可分问题">
# SVM 如何解决线性不可分问题

<span id = "为什么SVM的分类结果仅依赖于支持向量?">
# 为什么SVM的分类结果仅依赖于支持向量?

百机p53

<span id = "如何选取核函数?">
# 如何选取核函数?
最常用的是线性核与高斯核, 也就是 Linear 核与 RBF 核. 一般情况下 RBF 效果不会差于 Linear, 但是时间上 RBF 会耗费更多.
- Linear 核: 主要用于线性可分的情形. 参数少, 速度快, 对于一般数据, 分类效果已经很理想了.
- RBF 核: 主要用于线性不可分的情况. 参数多, 分类结果非常依赖于参数. 有很多人是通过训练数据的交叉验证来寻找合适的参数, 不过这个过程比较耗时. 个人体会是: 使用 libsvm, 默认参数, RBF 核比 Linear 核效果稍差. 通过进行大量参数的尝试, 一般能找到比 linear 核更好的效果. 至于到底该采用哪种核, 要根据具体问题和数据分析, 需要多尝试不同核以及不同参数. 如果特征提取的好, 包含的信息量足够大, 很多问题都是线性可分的. 当然, 如果有足够的时间去寻找合适的 RBF 核参数, 应该能取得更好的效果.

吴恩达的观点:
1. 如果 Feature 的数量很大, 跟样本数量差不多, 这时候可以使用 LR 或者是 Linear Kernel 的 SVM. (因为核函数需要计算内积, 两两样本都得算, 所以样本过多的话时间消耗太大, 很明显高斯核比线性核复杂的多)
2. 如果 Feature 的数量比较小, 样本数量一般, 不算大也不算小, 就选用 SVM + Gaussian Kernel
3. 如果 Feature 的数量比较小, 而样本数量比较多, 就需要手工添加一些 feature, 使之变成第一种情况.

<span id = "为什么说高斯核函数将原始特征空间映射成了无限维空间?">
# 为什么说高斯核函数将原始特征空间映射成了无限维空间?

https://blog.csdn.net/lin_limin/article/details/81135754

<span id = "核函数中不同参数的影响">
# 核函数中不同参数的影响

https://blog.csdn.net/lin_limin/article/details/81135754

https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247484495&idx=1&sn=4f3a6ce21cdd1a048e402ed05c9ead91&chksm=fdb699d8cac110ce53f4fc5e417e107f839059cb76d3cbf640c6f56620f90f8fb4e7f6ee02f9&scene=21#wechat_redirect

<span id = "既然深度学习技术性能表现已经全面超越 SVM, SVM 还有存在的必要吗?">
# 既然深度学习技术性能表现已经全面超越 SVM, SVM 还有存在的必要吗?


# Reference
[1] https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247483937&idx=1&sn=84a5acf12e96727b13fd7d456c414c12&chksm=fdb69fb6cac116a02dc68d948958ee731a4ae2b6c3d81196822b665224d9dab21d0f2fccb329&scene=21#wechat_redirect

[2] 西瓜书

[3] http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html


https://zhuanlan.zhihu.com/p/29212107
