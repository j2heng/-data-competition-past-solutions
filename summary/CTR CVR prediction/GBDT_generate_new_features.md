## Generate new features with GBDT

[reference: facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)

Overall idea : First train the GBDT model with existing features, then use the tree learned by the GBDT model to construct new features, and finally add these new features to the original features to train the model together. 

We treat each individual tree as a categorical feature that takes as value the index of the leaf an instance ends up falling in. We use 1-
of-K coding of this type of features.

__When a sample passes through a tree and falls in a leaf node, the value of this leaf node is set to 1 in the new feature vector, and the values of other leaf nodes of the same tree are set to 0__



For example, consider the boosted tree model in the figure below with 2 subtrees, where the first subtree has 3 leafs and the second 2 leafs. If an instance ends up in leaf 2 in the first subtree and leaf 1 in second subtree, the overall new feature vector will be the binary vector [0, 1, 0, 1, 0], where the first 3 entries correspond to the leaves of the first subtree and last 2 to those of the second subtree. 

![alt text](https://github.com/j2heng/collection-data-competition-past-solutions/blob/master/summary/CTR%20CVR%20prediction/image/fb_gbdt1.png)
