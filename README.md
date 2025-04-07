**这个仓库是自己在了解遥感变化检测相关知识的过程中自己复现的完整的任务流程，包括数据的可视化和一些基本任务逻辑的实现**
性能评估和数据集划分的方法参考了BIT这篇论文，[代码链接](https://github.com/justchenhao/BIT_CD/blob/master/main_cd.py)
由于笔记本算力有限，只复现了LEVIR数据集的数据，根据BIT论文中提到的模型参数配置<br>
![image](https://github.com/user-attachments/assets/869ab2f9-d18e-49b1-bf92-e2435277f230)

以下是我自己复现的对应结果<br>

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eb4ae54a-aa2a-49c7-a6e3-92e0db27db64" alt="MF1" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/d0099f95-31ed-4ebd-b8f3-2352521376ca" alt="MIOU" width="100%"></td>
  </tr>
</table>

我使用了以上相同配置的模型，在使用单独的交叉熵损失和Adam优化器在20个epoch内得到了性能与论文中的差距已经不大，相信在更好的显卡上加大batchsize和调优参数基本上可以复现原论文的性能，也说明这个代码实现的逻辑是没问题的

## 如果有任何问题请在issue区告诉我

但是很大地区别于之前的代码，我尽量使用了模块化和函数式编程使得每一个模块尽量的独立方便后续的更改

需要注意的点有如下：
1. 所有模型的训练基类被写在了trainer/Trainer.py中，也就是说如果有一个nn.Module的模型是可以完全继承这个Trainer更换掉model参数作为一个新的trainer来使用的
2. 基类的Trainer实现了最基本的数据加载，优化器配置和数据处理需要使用的collate_fn，如果你使用的Dataset返回的类型也是非list，dict类型的，需要在新的Trainer覆盖掉对应的collate_fn方法
3. 有关数据集的划分，参考_dataset_/crop_dataset.py对数据集进行划分
4. 使用了tensorbroad进行了数据可视化对应的训练生成文件在tensorbroad_log文件夹中
5. 保存训练输出日志文件对应的在result和logs中

## 数据集的划分
| 数据集    | Train              | Val  |
|--------|--------------------|------|
| LEVIR  | 7120               | 1024 |

<br>
所有的代码都是在Windows 11平台上进行测试，GPU为RTX 4060 Laptop (8G) 具体环境配置见requirement.txt

由于算力有限，我只使用了resnet18来对两个数据集进行的训练，参数也是非最优的，牺牲部分精度节省显存使用了半精度训练
<br>



