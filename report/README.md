# 文件清单
## 第一部分（位于 p1 文件夹下）

- A.mat：测试集的分类结果
- LR_first_order.m：梯度下降法优化LR算法主程序
- LR_first_second.m：牛顿法优化LR算法主程序
- LR_GD.m：梯度下降法具体实现部分
- LR_Newton.m：牛顿法具体实现部分
- LR.m：使用牛顿法得到的参数实现对任意输入得到输出的程序
- LR_first_order.mat：梯度下降法训练得到的参数
- LR_second_order.mat：牛顿法训练得到的参数
- Calc：sigmod函数实现


## 第二部分（位于 p2 文件夹下）
- B.npy: 测试集的分类结果
- B_feat.npy: 测试集的特征
- extract_feature.py: 提取特征的代码
- requirements.txt：依赖要求
- SVM.py: 题目所要求的分类函数
- svm_test.py: 利用训练集交叉验证代码
- train_feats.npy: 训练集的特征

## 第三部分（位于 p3 文件夹下）
- common文件夹：DataLoader、预处理和网络定义
- analyse.py：分析网络复杂度
- audio_test_feat.npy：已提取的测试集特征
- audio_train_feat.npy：已提取的训练集特征
- C.npy：模型在测试集上运行的结果
- debate_weights.pkl：训练好的模型参数
- requirements.txt：依赖要求
- test.py：在测试集上应用训练好的模型参数，得到测试集结果
- train_audio.py：使用音频特征进行训练
- train_image.py：使用图像特征进行训练 



# 环境搭建

## 第二部分

```shell
pip3 install -r requirements.txt --user
```



## 第三部分

```bash
pip3 install -r requirements.txt --user --find-links https://download.pytorch.org/whl/torch_stable.html
```



# 运行方法

## 第一部分

> 注：此部分的 `LR` 采用的是牛顿法的版本。 

调用 `LR.m` 中定义的 `LR(x)` 函数即可，`x` 为 n*13 维特征，返回值 `y` 为 n 维 0-1 向量的分类结果 。

## 第二部分

调用 `SVM.py` 中定义的 `SVM(x)` 函数即可，`x` 为 n*k 维特征，返回值 `y` 为 n 维 0-1 向量的分类结果 。

## 第三部分

训练

```shell
python3 ./train_audio.py ./dataset/train ./dataset/test
```

测试

```shell
python3 ./test.py ./dataset/train ./dataset/test
```

