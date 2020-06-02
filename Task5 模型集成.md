# Datawhale 零基础入门CV赛事-Task5 模型集成
## 1.集成学习
集成方法是将几种机器学习技术组合成一个预测模型的元算法，以达到减小方差（bagging）、偏差（boosting）或改进预测（stacking）的效果。

集成学习在各个规模的数据集上都有很好的策略。

**数据集大：** 划分成多个小数据集，学习多个模型进行组合

**数据集小：** 利用Bootstrap方法进行抽样，得到多个数据集，分别训练多个模型再进行组合首先将PPM

## 2.深度学习中的集成学习
### 2.1 Drop Out
在每个训练批次中，通过随机让一部分的节点停止工作。同时在预测的过程中让所有的节点都其作用。

使用Drop Out可以明显地减少过拟合现象。这种方式可以减少特征检测器（隐层节点）间的相互作用，检测器相互作用是指某些检测器依赖其他检测器才能发挥作用。

让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征

示例代码如下，其中nn.Dropout(0.25)即定义了dropout层，0.25即有25%的神经元会被dropout。
```
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
```

### 2.2 TTA
TTA即测试集数据扩增，是一种常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，对同一个样本预测三次，然后对三次结果进行平均。

代码如下：
```
def predict(test_loader, model, tta=10):
   model.eval()
   test_pred_tta = None
   # TTA 次数
   for _ in range(tta):
       test_pred = []
   
       with torch.no_grad():
           for i, (input, target) in enumerate(test_loader):
               c0, c1, c2, c3, c4, c5 = model(data[0])
               output = np.concatenate([c0.data.numpy(), c1.data.numpy(),
                  c2.data.numpy(), c3.data.numpy(),
                  c4.data.numpy(), c5.data.numpy()], axis=1)
               test_pred.append(output)
       
       test_pred = np.vstack(test_pred)
       if test_pred_tta is None:
           test_pred_tta = test_pred
       else:
           test_pred_tta += test_pred
   
   return test_pred_tta
```

## 2.3 Snapshot

深度神经网络模型复杂的解空间中存在非常多的局部最优解，经典的SGD方法只能让网络模型收敛到其中一个局部最优解，snapshot ensemble 通过循环调整网络学习率(cyclic learning rate schedule)使网络依次收敛到不同的局部最优解。

# 3.结果后处理
以目标检测中的**NMS**，即非极大值抑制为例

按照分类概率排序，概率最高的框作为候选框，其它所有与它的IOU高于一个阈值（通过人工指定的超参数）的框其概率被置为0。然后在剩余的框里寻找概率第二大的框。依次类推。

该方法应用非常广泛，但存在以下两点缺点：（1）NMS算法需要一个超参，超参的设计需要人工调试；（2）NMS会将相邻的两个大概率目标框去掉一个，造成漏检。

```
def py_cpu_nms(dets, thresh): 
  """Pure Python NMS baseline.""" 
	  #x1、y1、x2、y2、以及score赋值 
	  x1 = dets[:, 0] 
	  y1 = dets[:, 1] 
	  x2 = dets[:, 2] 
	  y2 = dets[:, 3] 
	  scores = dets[:, 4] 
	  #每一个检测框的面积 
	  areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
	  #按照score置信度降序排序 
	  order = scores.argsort()[::-1] 
	  keep = [] #保留的结果框集合 
	  while order.size > 0: 
	    i = order[0] keep.append(i) #保留该类剩余box中得分最高的一个 
	    #得到相交区域,左上及右下 
	    xx1 = np.maximum(x1[i], x1[order[1:]]) 
	    yy1 = np.maximum(y1[i], y1[order[1:]]) 
	    xx2 = np.minimum(x2[i], x2[order[1:]]) 
	    yy2 = np.minimum(y2[i], y2[order[1:]]) 
	    #计算相交的面积,不重叠时面积为0 
	    w = np.maximum(0.0, xx2 - xx1 + 1) 
	    h = np.maximum(0.0, yy2 - yy1 + 1) 
	    inter = w * h #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
	    ovr = inter / (areas[i] + areas[order[1:]] - inter) 
	    #保留IoU小于阈值的box 
	    inds = np.where(ovr <= thresh)[0] 
	    order = order[inds + 1] 
	  return keep
```
