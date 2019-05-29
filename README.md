# MobileNetV3-SSD
- 用自己的数据训练一个MobileNetV3，并以此为backbone加入到SSD中训练。



### 第一部分

第一部分MobileNetV3的内容非常简单，大致思路：创建网络，载入预训练模型，fine-tune网络，数据预处理，训练。

- 参考[MobileNetV3](https://github.com/leaderj1001/MobileNetV3-Pytorch)完成网络创建，非常感谢。MobileNetV3有两种网络结构LARGE和SMALL，我选择了SMALL来进行实验，载入预训练的模型之后只需要更具自己数据的类别数量修改最后一层网络即可。

- 数据预处理也是常规的简单处理：

  ```python
  normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
  train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.ImageFolder(traindir, torchvision.transforms.Compose([
      torchvision.transforms.Resize(input_size),
      torchvision.transforms.RandomCrop(input_size, padding=4),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      normalize,
  ])),
      batch_size=batch_size,
      shuffle=True,
      num_workers=n_workers,
      pin_memory=True
  )
  
  test_loader = torch.utils.data.DataLoader(						           				torchvision.datasets.ImageFolder(testdir, torchvision.transforms.Compose([
      torchvision.transforms.Resize(re_size),
      torchvision.transforms.CenterCrop(input_size),
      torchvision.transforms.ToTensor(),
      normalize,
  ])),
      batch_size=batch_size,
      shuffle=False,
      num_workers=n_workers,
      pin_memory=True
  )
  ```

30类的数据大概95k的数据，每类取出300张图像用作测试，每个EPOCH的训练时间在10min左右(GTX1070)，设置了20EPOCH之内没有收敛则停止训练，所以大概在140个EPOCH之后停止。训练之后在测试中能够达到99.60的准确率。



### 第二部分

- SSD的部分大致思路：将MobileNetV3作为backbone放入到SSD中，因为MobileNetV3刚出来不久，这部分的内容需要自己编写，但是SSD和MobileNetV3的工作已经完成度很好了，所以也没有太大难度，参考[SSD](<https://github.com/qfgaohao/pytorch-ssd>)完成，非常感谢。需要注意的一点是在backbone中选取的第一层特征图位置在我的MobileNetV3中第9个MobileBlock中第一个conv中的经过h-swish激活之后的位置即可，这个的选择依据是每层的特征图大小，当然论文中也有提及，变化趋势是19-10-5-3-2-1。我贴出了一个

  > “For MobileNetV3-Small, C4 is the expansion layer of the 9-th bottleneck block.”

  至此便可以开始训练网络，正在训练过程中，之后有了结果我会展示出来，代码我先上传SSD网络的部分，全部内容等之后整理好再一齐上传。

- 训练需要使用VOC2007样式的数据进行训练，关于制作数据集的内容简单记录一下。