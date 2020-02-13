# 第二章

本章讲解了了基本的三个步骤，数据整理，模型应用和模型调整。  
我觉得讲的比较细致的还是数据整理，模型应用本质来说和数据整理是一件事情，api相似度也很高。  
模型调整，这里也就讲了利用参数来的调优。

## 数组分组

[官方的一个文档](https://devdocs.io/scikit_learn/modules/cross_validation#cross-validation):`sklearn.model_selection`,这个包下面提供的各种方法。  
[交叉验证的wiki](https://zh.wikipedia.org/wiki/%E4%BA%A4%E5%8F%89%E9%A9%97%E8%AD%89):又成为循环验证。这样也就明白了很多了。具体分类什么的，就不深究了。
- 简单分组：`train_test_split` 
- 交叉验证：简单的来说，就是分多少次。其实这个还是有点研究的。具体还是看文档吧。就是这样会比较好。
    - 如果是时间序列的化，就需要做些其他的了。
    - 书中了`StratifiedShuffleSplit`这个方案。说句实话我么有太搞懂这些区别。
    - 这里所有的分类。基本都是给index。然后你根据index自己去分。
    - 分层，可以理解为分组

## 数据准备

### pandas

pandas其实也就那么几个drop方法来做数据整理。

### sklearn

最后用的应该是**pipline**。叫做管道，本质还是流一类的变成方式。核心方法，无非就是下面几个。
- `fit`:自动调整参数
- `transform`: 清洗数据
- `fit_transform`: 上面两个方法的联动。
这些类，设计有点类似于装饰这模式。即pipeline是上面三个主要方法。组成pipeline的组件，对外也是上面三个。  

## 注意点

1. 貌似从数据上来说，其处理的主要还是*numpy*，对*pandas*支持的不是很好，很多地方都要转。
2. `0.20.0`是一个大版本，有很多地方书中和这个版本，有很大不同。例子的代码还是看[github上的比较好](https://github.com/ageron/handson-ml)

- imputer: 估算器，书中的例子只是一个中位数估算。
    - `statistics_`: 获得估算的值
    - 大于*0.20.0*之后，名字已经改成了`SimpleImputer`了。
- 处理文本： 书中介绍的思路，本质来说，就是依据文本进行分类。然后通过数字来代表这个类型。
    - `OneHotEncoder`的改成二进制的根本原因，还是为了防止代表类别的数字产生了关系。  
    比如如果1，代表类型A，2代表B，3代表C。因为2和1之间的距离比1和3之间小。那么A和B的关系，就比A和C要进。这个在与大多数场合是错误的。所以才有了这个概念
    - 在`0.20.0`中，`OneHotEncoder`已经可以直接完成。具体看下面的代码
        - 这里貌似一口气只能用一行。。。 
```python
  one_hot_encoder = OneHotEncoder()
  one_hot_cat = one_hot_encoder.fit_transform(data[['ocean_proximity']])
  print(one_hot_cat.toarray())
```
- 自定义转换器。重写两个方法。`fit`和`transform`。还有两个基类。`BaseEstimator`和`TransformerMixin`
    - 基本上，就是处理一些特别的需求。
- 特征缩放： 书中就是讲的一些概念。然后给了两个常用的方案
    - `MinMaxScaler`: 归一化，把值改到0到1之间。这个范围，可以通过`feature_range`调整
    - `StandadScaler`: 标准化，受异常值影响更小。但是不能保证0到1
    - 只对训练集处理
- pipeline
    - 就是一个流式编程。内容，就是上面介绍的这些东西。
    - `FeatureUnion`:合并不同的pipeline
    - `ColumnTransformer`:但是对pandas的Dataframe需要每列不同处理的时候。

## 模型
这里没有太细讲模型。而只是很简单的说了一点点而已。以及一些优化模型的三个方向。

- 更换更加强有力的模型。
- 找寻更加好的特征
- 减少对模型的限制。 这点不是太明白。。

然后校验，则可以用sklearn的`mean_squared_error`。就是标准差那一套。

## 模型调整

模型调整，本质就是在那里做找寻一些最优化参数的调整。这里其实我有一个疑惑。因为之前看了另一部书，那个比较偏向于调整模型的。*这类型叫做超参吗？*  

但是从代码层面来说，我总结了一下子几点。

- pipeline：pipeline表示是一种**数据转换**，而模型的处理，也可以算是一种**数据转换**。也可以算成是其中的一环。
- 书中讲的两种参数寻找方式
    -  `GridSearchCV`: 在你给点的范围内寻找。Gird是网格的意思。就是网格的寻找，搜索的范围是颗粒的
    -  `RandomizedSearchCV`: 也是需要给定范围。只是这里的范围是连续的和颗粒度的。
- 所有搜索的返回值，也是一个pipeline。而其参数，也可以是一个pipeline
    - 传参的语法是支持嵌套的。每一集的分割是用两个下划线`__`。
    - 具体可以看exercise的例子。里面顶级是`preparation`,次级是`num`。。。以此类推。