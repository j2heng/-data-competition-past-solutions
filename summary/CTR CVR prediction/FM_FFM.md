# Why Factorization Machines (FM) and Field-Aware Factorization Machines (FFM)
FM and FFM solve the problem of considering __pairwise combination of features__. 

# How to consider pairwise combination of features
The most intuitive way is by using __polynomial model__. Here in case of pairwise combination, we use polynomial model of degree 2. 
However, there is one problem: after one-hot encoding, the matrix will be too sparse!

# Why sparsity is a problem
__The training of each parameter ```w_ij``` requires a large number of samples with both ```x_i``` and ```x_j``` being non-zero.__

Since the matrix is already sparse, fewer samples x_i, x_j will satisfy the condition of being both non-zero. 
Insufficient training samples can easily lead to inaccurate parameters, which will ultimately affect the performance of the model.

### _Factorization Machines(One parameter per feature)_
<img src="https://latex.codecogs.com/svg.latex?\Large&space;y(x)=w_0+\sum_{i=1}^{n}w_ix_i+\sum_{i=1}^{n}\sum_{j=i+1}^{n}w_{ij}x_ix_j" />

- where n is the number of features, xi is the value of the i-th feature, and w0, wi, and wij are parameters.
- there are a total number of n(n−1)/2 parameters for the pairwise features, and any two parameters are independent.
- the training of each parameter wij requires a large number of samples with both x_i and x_j being non-zero;
since the feature matrix is already sparse, the size of samples satisfying "both xi and xj are non-zero" will be very small.


# Solution 
__Use "Matrix Factorization" to solve the training problem of polynomial model/quadratic parameters.__

> In model-based collaborative filtering (CF), a rating matrix can be decomposed into a user matrix and an item matrix. 
> Each user and item can be represented by a vector. The dot product of user-X and item-A is the score of user X on the item A in the matrix.

Similarly, all quadratic parameter ```w_ij``` can form a symmetric matrix W, then the matrix can be decomposed into 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;W=V^TV" /> 
, the jth column of V is the vector of the j-th dimension feature. 
In other words, each parameter ```w_ij=⟨v_i,v_j⟩ ```, which is the core idea of the FM model

_**The problem of FM can be rewrote as**_ 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;y(x)=w_0+\sum_{i=1}^{n}w_ix_i+\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j" />

 ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) `
 Specifically, the coefficients of (x_h, x_i) and (x_i,x_j) are ⟨vh, vi⟩ and ⟨vi, vj⟩. 
 There is a common term vi between them. That is to say, all samples containing "non-zero pairwise features with xi" (i.e. xixj≠0) can be used to learn the vector vi, which largely avoids data sparsity. 
 While in the polynomial model, w_hi and w_ij are independent of each other.
 具体来说，xhxi 和 xixj的系数分别为 ⟨vh,vi⟩和 ⟨vi,vj⟩，它们之间有共同项 vi。也就是说，所有包含“xi的非零组合特征”（存在某个 j≠i，使得 xixj≠0）的样本都可以用来学习隐向量 vi，这很大程度上避免了数据稀疏性造成的影响。而在多项式模型中，whi 和 wij是相互独立的。
 `
 > 显而易见，y(x)是一个通用的拟合方程，可以采用不同的损失函数用于解决回归、二元分类等问题，比如MSE（Mean Square Error）求解回归问题，Hinge/Cross-Entropy求解分类问题，Logistic求解二元分类问题 (FM的输出经过sigmoid变换) 
通过公式(3)的等式，FM的二次项可以化简，其复杂度可以优化到 O(kn)。由此可见，FM可以在线性时间对新样本作出预测。

 > <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j=\frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f}x_i)^2-\sum_{i=1}^{n}v_{i,f}^2x_i^2)"/>
 > 其中，v_{j,f} 是隐向量 vj 的第 f 个元素
 
 # Model training
 We can use SGD (Stochastic Gradient Descent) to train the model.
 For FM, the gradients can be computed like this
 
 <img src="https://latex.codecogs.com/svg.latex?\frac{\partial}{\partial\theta}y(x)=\begin{cases}1&if\theta=w_0\\x_i&if\theta=w_i\\x_i\sum_{j=1}^nv_{j,f}x_j-v_{j,f}x_i^2&if\theta=v_{j,f}\end{cases}" />

### _Field-Aware Factorization Machines_

By introducing the concept of field, FFM put features of the same nature to the same field.

In FFM, each feature xi learns a vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;v_{i,f_j}" /> 
for each field f_j of other features. Therefore, the vector is not only related to the feature, but also related to the field.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;y(x)=w_0+\sum_{i=1}^{n}w_ix_i+\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i,f_j},v_{j,f_i}>x_ix_j" />

![#f03c15](https://placehold.it/15/f03c15/000000?text=+) `
FM: Vi 第i维特征的隐向量
FFM: Vi  becomes  [Vi,f1, Vi,f2, …, Vi,fj,…]   第i维特征的隐向量对每个field j都有个单独的隐向量
`
 # Model implementation
 Yu-Chin Juan implemented a [C++ version of the FFM model](https://github.com/ycjuan/libffm)
 
 This version of FFM ignores the constant and degree-1 terms.It uses logistic loss as loss function and L2 penalty term, so it can only be used for binary classification problems.
 
 Model uses SGD optimization, with Adaptive Learning Rates (similar to AdaGrad)
 > All features must be converted to "field_id:feat_id:value" format
 
 > 数值型的特征比较容易处理, categorical特征需要经过One-Hot编码成数值型
 
 > 在训练FFM的过程中，有许多小细节值得特别关注。
 > - 第一，样本归一化 [FFM默认处理]
 > - 第二，特征归一化 [把数据转换成libffm的数据格式前处理]
 > - 第三，省略零值特征 [把数据转换成libffm的数据格式前处理]
 
