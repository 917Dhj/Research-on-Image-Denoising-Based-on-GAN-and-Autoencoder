## 牛津猫狗分类模型

*这个模型的作用是给牛津猫狗数据集分类，分类为猫(0)和狗(1)。*

### 输入和输出

- **输入**：输入尺寸为[32, 3, 256, 256]的RGB彩色图像张量
- **输出**：输出一个尺寸为[1]的张量，输出经过sigmoid激活函数激活后就得到了模型预测的为“1”的概率

### 模型架构

因为分类模型的目的只是评估其他图像去噪模型的去噪效果，所以我们就先用了一个预训练的模型做了一些修改。采用的模型是**ResNet-50**模型，然后将最后一层替换成了一个新的线性层，以匹配我们需要的输出。

```python
# 加载预训练的ResNet-50模型
classifier = models.resnet50(pretrained=True)
classifier.fc = nn.Linear(classifier.fc.in_features, 1)  # 修改为二分类
```

### 损失函数和优化器

- **损失函数**：二元交叉熵`BCE`
- **优化器**：Adam优化器，学习率=0.0001
- **学习率调度器**：每5个epoch将学习率降低一半

```python
import torch.optim as optim

# 损失函数和优化器
criterion = nn.BCELoss() # 使用二元交叉熵损失函数
# 使用 Adam 优化器
optimizer = optim.Adam(classifier.parameters(), lr=0.0001) 
# # 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch将学习率降低一半
```

### 模型性能

对于牛津宠物数据集的256×256原始图像进行分类，在测试集上可以达到：

- **准确率**：`98.58%`
- **AUC**：`0.9997`

---

## 改良版<u>卷积自编码器</u>

### 输入和输出

- 输入尺寸为[3, 256, 256]的RGB彩色图像张量
- 图像来自数据集**牛津宠物**
- 输出跟输入尺寸**相同**的彩色图像

### 模型架构

```python
import torch
import torch.nn as nn

# 256x256的输入 改进版！
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入图像通道为3 (RGB)，输出为32个特征图
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 添加批量归一化
            nn.MaxPool2d(2, 2),  # 下采样，图像尺寸减半 (256 -> 128)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64个特征图
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.MaxPool2d(2, 2),  # 下采样 (128 -> 64)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128个特征图
            nn.ReLU(),
            nn.BatchNorm2d(128),  # 批量归一化
            nn.MaxPool2d(2, 2)  # 下采样 (64 -> 32)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样，图像尺寸增大 (32 -> 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 批量归一化
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样 (64 -> 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 批量归一化
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样 (128 -> 256)
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0, 1]范围内
        )
    
    def forward(self, x):
        x = self.encoder(x)  # 编码器前向传播
        x = self.decoder(x)  # 解码器前向传播
        return x
```

### 损失函数和优化器

- **损失函数**：`MS_SSIM`和`MSE`损失加权，其中`MS_SSIM`的权重为`0.8`
- **优化器**：Adam优化器，学习率=0.0001
- **学习率调度器**：每5个epoch将学习率降低一半

```python
criterion = CombinedLoss(alpha=0.8)  # 将MS_SSIM_Loss 和 MSE损失加权 alpha是加权的权重，alpha 越大，MS-SSIM 损失的影响越大
# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 加入L2正则化
# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch将学习率降低一半
```

### 模型评估结果

#### 1. 基于测试集

- **测试集平均损失**：`Test Loss: 0.0371`

- **平均PSNR**：`Average PSNR: 23.5759 dB`

- **平均SSIM**：`Average SSIM: 0.6403`

#### 2. 基于肉眼

![image-20250223202348551](/Users/dinghongjing/Library/Application Support/typora-user-images/image-20250223202348551.png)

#### 3. 基于分类模型

**分类模型在<u>噪声图像</u>上的性能指标**：

- **准确率**: `68.20%`

- **AUC值**: `0.5377`

**分类模型在卷积自编码器<u>去噪后的图像</u>上的分类性能指标**：

- **准确率**：`78.08%`
- **AUC值**：`0.8362`

---

## 卷积自编码器结合GAN

*这个模型是以**卷积自编码器为生成器**，**卷积神经网络为判别器**，构造的一个用于256×256图像去噪的**生成对抗网络***

### 1. 生成器

#### 输入&输出

- 输入尺寸为[3, 256, 256]的RGB彩色图像张量
- 输出跟输入尺寸**相同**的彩色图像

#### 生成器架构

生成器模型用的就是之前用过的改良版的卷积自编码器，因为我们的目的是将它与GAN结合，试图涨点。

```python
import torch
import torch.nn as nn

# 改进版 卷积自编码器作为 生成器
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入图像通道为3 (RGB)，输出为32个特征图
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 添加批量归一化
            nn.MaxPool2d(2, 2),  # 下采样，图像尺寸减半 (256 -> 128)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64个特征图
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 批量归一化
            nn.MaxPool2d(2, 2),  # 下采样 (128 -> 64)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128个特征图
            nn.ReLU(),
            nn.BatchNorm2d(128),  # 批量归一化
            nn.MaxPool2d(2, 2)  # 下采样 (64 -> 32)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样，图像尺寸增大 (32 -> 64)
            nn.ReLU(),
            nn.BatchNorm2d(64),  # 批量归一化
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样 (64 -> 128)
            nn.ReLU(),
            nn.BatchNorm2d(32),  # 批量归一化
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 上采样 (128 -> 256)
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0, 1]范围内
        )
    
    def forward(self, x):
        x = self.encoder(x)  # 编码器前向传播
        x = self.decoder(x)  # 解码器前向传播
        return x
```

#### 损失函数和优化器

**生成器的损失是两个损失的和**：

- **==自编码器的重建损失==**：用的是跟之前一样的`MS-SSIM`损失与`MSE`损失加权的方式，权重为`0.8`
- **==生成器的对抗损失==**：生成器希望生成尽可能真实的图像，目的是让判别器判断这些图像为真实图像。判别器的输出是一个介于0到1之间的值，表示图像是否真实。
  - 在代码中`d(fake_images)`就是走了判别器的前向传播过程，它返回的结果就是判别器对于`fake_images`也就是生成器生成的假图像的判别结果。
  - `(d(fake_images) - 1) ** 2`就是计算**判别器的输出与1之间的距离的平方**，也就是**判别器的输出值与生成器期望的值1之前的方差**，对于生成器来说，这个方差**越小越好**，这就是生成器的对抗损失，它是以**判别器的输出为基础计算的**。

- 将**重建损失和对抗损失相加**，就是**生成器的损失**。这样就使得生成器在训练中即能考虑到图像的**重建质量**，也能考虑到**生成让判别器更难判别的图像**。

```python
import torch.optim as optim

# 损失函数和优化器
criterion = CombinedLoss(alpha=0.8)  # 将MS_SSIM_Loss 和 MSE损失加权 alpha是加权的权重，alpha 越大，MS-SSIM 损失的影响越大

# 生成器的损失函数
def generator_loss(d, real_images, fake_images, criterion):
    # 计算自编码器的重建损失，即MS-SSIM损失
    recon_loss = criterion(fake_images, real_images)
    # 计算生成器的对抗损失
    adv_loss = torch.mean((d(fake_images) - 1) ** 2)  # 生成器希望判别器认为生成的图像是真的
    return recon_loss + adv_loss

# 生成器 使用 Adam 优化器 学习率为 0.0001
optimizer_gen = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999)) 

# 生成器 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer_gen, step_size=5, gamma=0.5)  # 每5个epoch将学习率降低一半
```

### 2. 判别器

#### 输入&输出

- 判别器的输入是尺寸为[3, 256, 256]的彩色图像张量，这个图像要么是真实的原始图像，要么是生成器生成的假图像。

- 判别器的输出是尺寸为[1]的张量，表示输入的图像为真的概率，是一个0-1之间的数值。

#### 判别器架构

判别器是一个普通的卷积神经网络

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512*16*16, 1),  # 适应输入尺寸
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
```

#### 损失函数和优化器

*判别器的目标是：对真实图像输出接近1，对假图像输出接近0*

**判别器的损失是真实图像的损失和假图像的损失的平均值**：

- **真实图像的损失**：就是判别器的输出与1的方差。这个损失越小，判别器的输出就能越接近1。
- **假图像的损失**：就是判别器的输出与0的方差，这个损失越小，判别器的输出就越接近0。
- 判别器的训练目标就是让这两个损失都**越小越好**，所以将它们**取平均**作为**最终的损失**是个很好的办法。

```python
# 判别器的损失函数
def discriminator_loss(d, real_images, fake_images):
    # 真实图像的损失：判别器应把真实图像标记为1
    real_loss = torch.mean((d(real_images) - 1) ** 2)  # 判别器输出接近1
    # 生成图像的损失：判别器应把生成图像标记为0
    fake_loss = torch.mean(d(fake_images) ** 2)  # 判别器输出接近0
    return (real_loss + fake_loss) / 2  # 真实图像和生成图像的损失平均

# 判别器 使用 Adam 优化器 学习率为 0.0001
optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 生成器 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer_gen, step_size=5, gamma=0.5)  # 每5个epoch将学习率降低一半
```

### 3. 模型评估结果

#### 3.1 测试集表现

- **测试集平均损失**：`Test Loss: 0.0359`

- **平均PSNR**：`Average PSNR: 23.9254 dB`

- **平均SSIM**：`Average SSIM: 0.6505`

#### 3.2 肉眼评估

![image-20250223215742025](/Users/dinghongjing/Library/Application Support/typora-user-images/image-20250223215742025.png)

#### 3.3 分类表现

**分类模型在<u>噪声图像</u>上的性能指标**：

- **准确率**: `67.66%`

- **AUC值**: `0.5699`

**分类模型在卷积自编码器<u>去噪后的图像</u>上的分类性能指标**：

- **准确率**：`79.09%`
- **AUC值**：`0.8452`

## 模型性能对比

|                                | 改良版卷积自编码器 | 卷积自编码器结合GAN |
| :----------------------------: | :----------------: | :-----------------: |
| **测试集平均损失（越小越好）** |       0.0371       |       0.0359        |
|    **平均PSNR（越大越好）**    |     23.5759 dB     |     23.9254 dB      |
|    **平均SSIM（越大越好）**    |       0.6403       |       0.6505        |
|   **分类准确率（越大越好）**   |       78.08%       |       79.09%        |
|    **分类AUC（越大越好）**     |       0.8362       |       0.8452        |

