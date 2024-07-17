# Variational Auto-Encoder (VAE)

因为DDPM(Denoising Diffusion Probabilistic Models)在逆扩散的过程中的概率推断采用了和VAE类似的变分推断，因此有必要对VAE做一定的了解

Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)

参考资料 （https://blog.csdn.net/weixin_42491648/article/details/132384913）

## 变分下界的推导以及随机变分推断(stochastic variational inference)

自编码器(Autoencoder,AE)通过构建编码器实现数据的压缩和重建。变分自编码器(VAE)在编码器的输出上引入随机性，通过概率的形式生成潜在空间表达，通过解码器从随机分布的编码器输出上重建原数据的概率预测。

VAE的目标是优化生成模型(the generative model)即Decoder的参数$\theta$,使得模型的输出概率最大。即是对数似然函数$logp_{\theta}(x)$最大

$p_{\theta}(z|x)$表示潜在空间变量$z$对原数据$x$的后验概率，因为实际的后验概率难以获得，文中使用识别模型$q_{\phi}(z|x)$对后验概率进行近似表示,$q_{\phi}(z|x)$代表VAE的encoder部分即由输入的$x$获得潜在空间表达$z$的过程. ，原文“let us intorduce a recognition model $q_{\phi}(z|x)$: an approximation to the intractable true posterior $p_{\theta}(z|x)$.” 

$logp_{\theta}(x) = log \int p_{\theta}(x,z)dz = log \int p_{\theta}(z) p_{\theta}(x|z)dz = log \int p_{\theta}(z|x) p_{\theta}(x)dz$

进一步由于VAE假设了$p_{\theta}(z)$潜在空间表达$z$的先验分布为标准正态分布，因此潜在变量$z$不依赖输入变量$x$。这里能够使用$q_{\phi}(z)$代替$p_{\theta}(z)$，原文“We assume an approximate posterior in the $q_{\phi}(z|x)$, but please note that the technique can be applied to the case $q_{\phi}(z)$, i.e. where we do not condition on $x$”。上式转化为

$logp_{\theta}(x) = log \int q_{\phi}(z) p_{\theta}(x)dz$ 

在VAE中选择标准正态分布表示潜在空间表达$z$，因此$z$,$x$相互独立，且$\int q_{\phi}(z)dz = 1, q_{\phi}(z|x)=q_{\phi}(z)$, 

$logp_{\theta}(x) = log \int q_{\phi}(z) p_{\theta}(x)dz = \int q_{\phi}(z) logp_{\theta}(x)dz = \int q_{\phi}(z|x) logp_{\theta}(x)dz$

由此$logp_{\theta}(x)$可变换为：

$\begin{aligned}
logp_{\theta}(x) &= \int q_{\phi}(z|x)logp_{\theta}(x)dz\\
&= \int q_{\phi}(z|x)log\frac{p_{\theta}(z,x)}{p_{\theta}(z|x)}dz \\
&= \int q_{\phi}(z|x)log\left (\frac{p_{\theta}(z,x)}{p_{\theta}(z|x)} \frac{q_{\phi}(z|x)}{q_{\phi}(z|x)}\right)dz \\
&= \int q_{\phi}(z|x)log\frac{p_{\theta}(z,x)}{q_{\phi}(z|x)}dz + \int q_{\phi}(z|x)log\frac{q_{\phi}(z|x)}{p_{\theta}(z|x)}dz \\
&= \int q_{\phi}(z|x)log\frac{p_{\theta}(x|z)p_{\theta}(z)}{q_{\phi}(z|x)}dz + D_{KL}(q_{\phi}(z|x)||p_{\theta}(z|x)) \\
&= \mathcal{L}(\theta,\phi;x) + D_{KL}(q_{\phi}(z|x)||p_{\theta}(z|x))
\end{aligned}$

上式中KL散度项表示识别模型$q_{\phi}(z|x)$对真实后验概率$p_{\theta}(z|x)$的近似程度。

KL散度的非负性证明：

$D_{KL}(q|p) = \sum_{i} q log\frac{q}{p} = \sum_{i} p \frac{q}{p} log\frac{q}{p}$ 

令$x = \frac{q}{p}, D_{KL}(q|p) = \sum_{i}pxlogx$

令$f(x)=xlogx, f'(x)=1+logx, f''(x)=\frac{1}{x}, \because f''(x)>0, \therefore f(x)是凸函数, \therefore E(f(x))\ge f(E(x))$

$\sum_{i}pxlogx \ge \sum_{i}pxlog(\sum_{i}px) = \sum_{i}pxlog1 = 0$

所以可得$D_{KL}(q|p) \ge 0$，$logp_{\theta}(x) \ge \mathcal{L}(\theta,\phi;x)$, $\mathcal{L}(\theta,\phi;x)$ 是对$logp_{\theta}(x)$的估计下界ELBO(Evidence lower Bound)

进一步对变分下界进行化简

$\begin{aligned}
\mathcal{L}(\theta,\phi;x) &= \int q_{\phi}(z|x)log\frac{p_{\theta}(x|z)p_{\theta}(z)}{q_{\phi}(z|x)}dz \\
&= \int q_{\phi}(z|x)log\frac{p_{\theta}(z)}{q_{\phi}(z|x)}dz + \int q_{\phi}(z|x) log(p_{\theta}(x|z))dz \\
&= -\int q_{\phi}(z|x)log\frac{q_{\phi}(z|x)}{p_{\theta}(z)}dz + \int q_{\phi}(z|x) log(p_{\theta}(x|z))dz \\
&= -D_{KL}(q_{\phi}(z|x)||p_{\theta}(z)) + E_{q(x|z)}[log(p_{\theta}(x|z))]
\end{aligned}$

至此原问题最大化对数似然函数$logp_{\theta}(x),等同于最大化其ELBO,对于上式等同于最小化第一项即$q_{\phi}(z|x)$识别模型encoder生成的潜在变量对$p_{\theta}(z)$潜在变量先验分布的近似能力，同时最大化第二项$E_{q(x|z)}[log(p_{\theta}(x|z))]$即通过生成模型decoder,从潜在变量$z$重建原数据$x$其结果和输入的$x$越接近。

VAE的目标是从满足正态分布$\mathcal{N}(0,1)$的潜在变量$z$恢复输入$x$.Encoder的输出$q_{\phi}(z|x)$服从高斯分布$\mathcal{N}(\mu,\sigma^2)$.将二者代入第一项KL散度中，可得：

$\begin{aligned}
D_{KL}(q_{\phi}(z|x)||p_{\theta}(z)) &= D_{KL}(\mathcal{N}(\mu,\sigma^2)|\mathcal{N}(0,1)) \\
&= \int \frac{1}{\sqrt{2\pi\sigma^2}} e^{\frac{-(x-\mu)^2}{2\sigma^2}}\left (log\frac{e^{\frac{-(x-\mu)^2}{2\sigma^2}}/\sqrt{2\pi\sigma^2}}{e^{\frac{-x^2}{2}}/\sqrt{2\pi}} \right )dx \\
&= \frac{1}{\sqrt{2\pi\sigma^2}} \int e^{\frac{-(x-\mu)^2}{2\sigma^2}}\left ( loge^{\frac{-(x-\mu)^2}{2\sigma^2}} - log e^{\frac{-x^2}{2}} - log \sqrt{\sigma^2}\right )dx \\
&= \frac{1}{\sqrt{2\pi\sigma^2}} \int e^{\frac{-(x-\mu)^2}{2\sigma^2}}\left ( -\frac{(x-\mu)^2}{2\sigma^2} + \frac{x^2}{2} - \frac{1}{2}log\sigma^2\right )dx \\
&= \frac{1}{2} \int \frac{1}{\sqrt{2\pi\sigma^2}} e^{\frac{-(x-\mu)^2}{2\sigma^2}}\left ( -log\sigma^2 + x^2 - \frac{(x-\mu)^2}{\sigma^2}\right )dx \\
&= \frac{1}{2} \left ( -log\sigma^2 + E[x^2] - 1\right ) = \frac{1}{2} \left ( -log\sigma^2 + \mu^2 + \sigma^2 - 1\right )
\end{aligned}$

对于ELBO中的第二项$E_{q(x|z)}[log(p_{\theta}(x|z))]$通过使用采样来近似表示

$E_{q(x|z)}[log(p_{\theta}(x|z))] = \frac{1}{L}\sum^L_{i=0}logp_{\theta}(x^{(i)}|z^{(i,l)})$

这里介绍一下逻辑回归的似然函数,潜在空间变量$z^{i}$通过生成模型$p_{\theta}(x^{i}|z^{i})$获得输出$y^{i}$近似输入$x^{i}$的分布

$p_{\theta}(x|z) = \prod_{i=1}^{n}[y_i^{x_i}(1-y_i)^{1-x_i}]$,标准的逻辑回归条件下$x_i$为0~1分布,

其对数似然函数$logp_{\theta}(x|z) = \sum_{i=1}^{n}[x_ilogy_i+(1-x_i)log(1-y_i)]$,可以从$x_i$为0~1分布的情况推广到$x_i \in (0,1)$之间的一般情况。

The code comes from https://github.com/AntixK/PyTorch-VAE/tree/master/models/vanilla_vae.py
```python
class VanillaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512] # 5次下采样 默认输入 64x64

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim) 
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3, kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input) # stride=2的卷积进行下采样构造潜在变量
        result = torch.flatten(result, start_dim=1) # 维度在这里展开 result: [N x C H W]

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result) # 生成潜在变量分布的均值和方差
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # 原文中使用FFN从隐变量z推导相应的x，但在这里由于主要对图像处理使用转值卷积，从小尺寸的feature map恢复到原始尺寸
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        # 回复输入的channel数
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu # 重构隐变量

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons，input，mu，log_var = args[:4]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)  # ELBO中的第二项，这里没有采用交叉熵的形式，但mse也没差同样能表示输出对输入的接近
        
        # ELBO的第一项，VAE使用N(0,1)的随机变量作为潜在变量，使用Q(z|x)的生成模型去逼近N(0,1)的正态分布
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) 

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, num_samples:int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim) # VAE使用N(0,1)正态分布生成潜在变量
        # 参考 https://blog.csdn.net/weixin_42491648/article/details/132384913 

        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
```

## Vector Quantized Variational AutoEncoder (VQ-VAE)

Paper: [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)

VQ-VAE通过离散化的潜在变量代替VAE使用连续高斯分布作为潜在变量,可以避免后验坍塌问题(潜在变量分布趋向于先验而使生成能力的多样性下降)，由可控的向量数量保证模型训练的稳定性,同时学习到更结构化和可解释的数据表示。

VQ-VAE中最好玩的地方就是Vector Quantization部分,通过stop gradient解决最近邻搜索不可导的问题。

The code comes from https://github.com/AntixK/PyTorch-VAE/tree/master/models/vq_vae.py
```python
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D) # self.K embed空间尺寸(包含的向量个数)， self.d向量长度
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K) # 初始化

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D] 展开方便计算

        # Compute L2 distance between latents and embedding weights
        # 计算encoder输出和embeding空间向量的距离，文中eq(1)
        # 按如下方法计算可以减少运算量和存储空间
        # 第一个sum时keepdim=True,因为加减运算是broadcastable
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        # encoding_inds：[BHW, 1], encoding_one_hot：[BHW, k],通过将encoding_ins距离最近的值作为索引将对应位置赋值1，构成最近距离的one-hot形式
        # 效果和torch.nn.functional.one_hot相同，但后者会返回一个新变量
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]， scatter_的参数顺序(axis,index,src)

        # Quantize the latents
        # 通过matmul将encoding_one_hot中对应id的向量赋给quantized_latents
        # 该步实现文中eq(2)
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        # 文中eq(3)的第二项和第三项，detach()实现sg(stop gradiant)操作
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents) 
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # detach()将tensor从模型中摘出来，下式做forward时值等于quantized_latents,bacward时梯度直接传递给latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
```