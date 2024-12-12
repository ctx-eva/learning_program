# Fundamental model of Diffuison

在扩散模型图像生成任务中两种基础的描述扩散过程的方法:1.基于对图像去噪的逆扩散过程。2.基于数据分布的对数概率梯度(Score function)的学习。

对于1由于其基础模型[Deep Unsuperviesd Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585)因为噪声设计等原因导致其计算复杂度高,模型稳定性差,所以选择更加简便的形式Denoising Diffusion Probabilistic Models (DDPM)。同样的原因对于2的原型[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600)选择Score-Based GEnerative Modeling Through Stochastic Differential Equations(Score-based SDE)来代替。

## Denoising Diffusion Probabilistic Models (DDPM)

Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)

因为DDPM在其推导中使用了负对数似然变分界，所以了解一下[变分推断(Variational inference)](./README_VAE.md)是有必要的。

原文对于DDPM反向过程的描述“the reverse process, and it is defined as a Markov chain with learned Gaussian transitions”

$p_{\theta}(x_{0:T}) = p_\theta(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t}), where \ p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1};\mu_{\theta}(x_t,t),\Sigma_{\theta}(x_t,t))$

上式中t步的reverse过程中,$\mu_{\theta}$和$\Sigma_{\theta}$是通过$t$时刻的状态$x_t$估计出来的分布参数。

前向过程的原文描述“the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1,\cdots,\beta_T$”。

$q_{\phi}(x_{1:T}|x_0)=\prod_{t-1}^Tq_{\phi}(x_t|x_{t-1}), where \ q_{\phi}(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$

DDPM中的扩散过程在维持信号和噪声的总方差不变的条件下，不断的缩小信噪比(SNR)。通过上式描述的过程可以从$x_0$中采样任意$t$的状态$x_t$,原文引入了标志$\alpha_t=1-\beta_t,\bar{\alpha}_t=\prod_{s=1}^T\alpha_s$,但对于$\alpha,\bar{\alpha},\beta$有太多的符号了，这里我采用和Denosing Diffusion Implicit Models(DDIM)中相同的形式只保留$\bar{\alpha}$并将其写作$\alpha$.由此t时刻的前向过程表示为：

$q_{\phi}(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{\frac{\alpha_t}{\alpha_{t-1}}}x_{t-1},\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right) I)$

$q_{\phi}(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\alpha_t}x_0,(1-\alpha_t)I)$

$q_{\phi}(x_t|x_0)$同时可表示为训练过程中对t步的采样过程

根据VAE的变分推断：

$logp_{\theta}(x) \ge \mathcal{L}(\theta,\phi;x) = \int q_{\phi}(z|x)log\frac{p_{\theta}(z,x)}{q_{\phi}(z|x)}dz$

将上式中的潜在变量$z$替换成扩散过程的所有中间变量$x_{1:T}$,令生成变量$x=x_0$,则上式可转化为扩散过程的变分下界，对其取对数取负，获得扩散过程的负对数上界

$-logp_{\theta}(x_0) \le \int -q_{\phi}(x_{1:T}|x_0)log\frac{p_{\theta}(x_{1:T},x_0)}{q_{\phi}(x_{1:T}|x_0)}dx_{1:T} = \int -q_{\phi}(x_{1:T}|x_0)log\frac{p_{\theta}(x_{0:T})}{q_{\phi}(x_{1:T}|x_0)}dx_{1:T}$

将reverse过程和diffuse过程代入上式

$-logp_{\theta}(x_0) \le \int -q_{\phi}(x_{1:T}|x_0)logp_\theta(x_T)\prod_{t=1}^T\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_t|x_{t-1})}dx_{1:T} = E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t=1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_t|x_{t-1})}\right]$

Appendix A中的推导,参考Deep Unsuperviesd Learning using Nonequilibrium Thermodynamics中Appendix B.3的代换“because the forward trajectory is a Markov process”,因为扩散过程中$x_t,x_{t-1}$都由$x_0$递推得到,因此可以做如下代换$q_{\phi}(x_t|x_{t-1}) = q_{\phi}(x_t|x_{t-1},x_0)$

$\begin{aligned}
-logp_{\theta}(x_0) \le & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t=1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_t|x_{t-1},x_0)}\right] \\
& \text{用t步对t-1步和0步的后验代替t步对t-1步的后验} \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_t|x_{t-1},x_0)} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
& \text{抽出t=1步的项} \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_t|x_{t-1},x_0)\frac{q_{\phi}(x_{t-1}|x_0)}{q_{\phi}(x_{t-1}|x_0)}\frac{q_{\phi}(x_t|x_0)}{q_{\phi}(x_t|x_0)}} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)\frac{q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)}} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)}\frac{q_{\phi}(x_{t-1}|x_0)}{q_{\phi}(x_t|x_0)} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
& \text{组合出t-1步对t步和0步的后验项,并配平} \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)} - \sum_{t \ge 1}^T log\frac{q_{\phi}(x_{t-1}|x_0)}{q_{\phi}(x_t|x_0)} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -logp_\theta(x_T) - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)} - log\frac{q_{\phi}(x_1|x_0)}{q_{\phi}(x_T|x_0)} - log\frac{p_{\theta}(x_0|x_1)}{q_{\phi}(x_1|x_0)}\right] \\
= & E_{q_{\phi}(x_{1:T}|x_0)}\left[ -log\frac{p_\theta(x_T)}{q_{\phi}(x_T|x_0)} - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)} - logp_{\theta}(x_0|x_1)\right] \\
& \text{log内相乘等于在外相加,抽出配平项,并对分子分母相消除,结果和第一第三项相合并} \\
= & \int q_{\phi}(x_{1:T}|x_0)\left[ -log\frac{p_\theta(x_T)}{q_{\phi}(x_T|x_0)} - \sum_{t \ge 1}^T log\frac{p_{\theta}(x_{t-1}|x_{t})}{q_{\phi}(x_{t-1}|x_t,x_0)} - logp_{\theta}(x_0|x_1)\right] dx_{1:T}\\
= & \int q_{\phi}(x_{1:T-1}|x_0)dx_{1:T-1}\left[q_{\phi}(x_T|x_0)log\frac{q_{\phi}(x_T|x_0)}{p_\theta(x_T)} \right] dx_T + \int q_{\phi}(x_{1:T,x\neq t-1}|x_0)dx_{1:T,x\neq t-1} \left[\sum_{t \ge 1}^T q_{\phi}(x_{t-1}|x_t)log\frac{q_{\phi}(x_{t-1}|x_t,x_0)}{p_{\theta}(x_{t-1}|x_{t})}\right]dx_{t-1} - \int q_{\phi}(x_{1:T-1}|x_0) logp_{\theta}(x_0|x_1) dx_{1:T-1} \\
& \text{将求平均展开，分配给各项，对后验概率需要进一步展开} \\
= & \int \left[q_{\phi}(x_T|x_0)log\frac{q_{\phi}(x_T|x_0)}{p_\theta(x_T)} \right] dx_T + \sum_{t \ge 1}^T \int \left[ q_{\phi}(x_{t-1}|x_t)log\frac{q_{\phi}(x_{t-1}|x_t,x_0)}{p_{\theta}(x_{t-1}|x_{t})}\right]dx_{t-1} - \int q_{\phi}(x_{1:T-1}|x_0) logp_{\theta}(x_0|x_1) dx_{1:T-1} \\
& \text{后验概率对其先验项的积分等于一} \\
= & D_{KL}\left[q_{\phi}(x_T|x_0)|p_\theta(x_T)\right] + \sum_{t \ge 1}^T D_{KL}\left[q_{\phi}(x_{t-1}|x_t,x_0)|p_{\theta}(x_{t-1}|x_{t})\right] + E_{q_{\phi}(x_{1:T}|x_0)} logp_{\theta}(x_0|x_1)\\ 
& \text{将第一二项表示为KL散度的形式}
\end{aligned}$

以上获得扩散过程的负对数上界的推导结果，即为原文中的式(5)

计算后验概率$q_{\phi}(x_{t-1}|x_t,x_0)$的表达式,即原文中式(6),式(7)。

$q_{\phi}(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_tI)$

$\tilde{\mu}_t(x_t,x_0) = \frac{\sqrt{\alpha_{t-1}}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{1-\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\alpha_{t-1})}{\sqrt{\alpha_{t-1}(1-\alpha_t)}}x_t, and \  \tilde{\beta}_t = \frac{1-\alpha_{t-1}}{1-\alpha_t}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}} \right)$

原文“all KL divergences are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions”。说实话即便查询了资料我也没看懂这块分解是怎么实现的,这里只对于上式做一个验证性的校对。

将$x_0 = \sqrt{\frac{1}{\alpha_t}}x_t - \sqrt{\frac{1 - \alpha_{t-1}}{\alpha_t}}I$代入式(6),(7)

$\begin{aligned}
\tilde{\mu}_t(x_t,x_0) = & \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{1-\alpha_t}x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}I + \frac{\sqrt{\alpha_t}(1-\alpha_{t-1})}{\sqrt{\alpha_{t-1}(1-\alpha_t)}}x_t \\
= & \frac{\alpha_{t-1}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right) + \alpha_t(1 - \alpha_{t-1})}{\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}(1-\alpha_t)} x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}I \\
= & \frac{\alpha_{t-1} - \alpha_t + \alpha_t - \alpha_t\alpha_{t-1}}{\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}(1-\alpha_t)} x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}I \\
= & \frac{\alpha_{t-1}(1 - \alpha_t)}{\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}(1-\alpha_t)} x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}I \\
= & \sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}I 
\end{aligned}$

$E[q_{\phi}(x_{t-1}|x_t,x_0)] = E[\tilde{\mu}_t(x_t,x_0)] = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t$

$D[q_{\phi}(x_{t-1}|x_t,x_0)] = D[\tilde{\mu}_t(x_t,x_0)] + D[\tilde{\beta}_t] 
= \frac{\alpha_{t-1}}{\alpha_{t}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)^2}{1-\alpha_t} + \frac{1-\alpha_{t-1}}{1-\alpha_t}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}} \right) 
= \frac{\alpha_{t-1} - \alpha_t + \alpha_t - \alpha_t\alpha_{t-1}}{\alpha_t(1-\alpha_t)}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}} \right) 
= \frac{\alpha_{t-1}}{\alpha_t}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}} \right)$

$q_{\phi}(x_{t-1}|x_t,x_0)$的均值和方差和从$q_{\phi}(x_t|x_{t-1})$解算得到$p_{\theta}(x_t|x_{t-1})$的一致。

对负对数变分上界的优化转化为对式(5)中第二项$\sum_{t \ge 1}^T D_{KL}\left[q_{\phi}(x_{t-1}|x_t,x_0)|p_{\theta}(x_{t-1}|x_{t})\right]$的优化,将$q_{\phi}(x_{t-1}|x_t,x_0)$和$p_{\theta}(x_t|x_{t-1})$二者的高斯分布带入。

$\begin{aligned}
L_{t-1} = & D_{KL}\left[q_{\phi}(x_{t-1}|x_t,x_0)|p_{\theta}(x_{t-1}|x_{t})\right] = E_q \left[ log\frac{q_{\phi}(x_{t-1}|x_t,x_0)}{p_{\theta}(x_{t-1}|x_{t})} \right] = E_q \left[ log \frac{1}{\sqrt{2\pi}\sigma_1}e^{-\frac{[x-\mu_t(x_t,x_0)]^2}{2\sigma_1^2}} - log \frac{1}{\sqrt{2\pi}\sigma_2}e^{-\frac{[x-\mu_{\theta}(x_t,t)]^2}{2\sigma_2^2}}\right] \\
& \text{已经证明KL散度的两项的方差一致} \\
= & E_q \left[\frac{1}{2\sigma_t^2}[x-\mu_t(x_t,x_0)]^2 - \frac{1}{2\sigma_t^2}[x-\mu_{\theta}(x_t,t)]^2 \right] = E_q \left[\frac{1}{2\sigma_t^2}\left\|\tilde{\mu}_t(x_t,x_0) - \mu_{\theta}(x_t,t) \right\|^2 \right] \\
& \text{根据t步对0步的后验,将用t步表示的0步代入,并联系原文式(7),其结果以由之前的验证部分获得} \\
= & E_{x_0,\varepsilon} \left[\frac{1}{2\sigma_t^2}\left\| \sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t - \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}\varepsilon - \mu_{\theta}(x_t,t) \right\|^2 \right]
\end{aligned}$

上式可得

$\mu_{\theta}(x_t,t) = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}} \left[ x_t - \frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}\varepsilon \right]$

将噪声$\varepsilon$作为估计,写作模型函数$\varepsilon_{\theta}(x_t,t)$。将$\mu_{\theta}(x_t,t)$代回原文MarKov递推$x_{t-1}\sim p_{\theta}(x_{t-1}|x_t)$,得到DDPM的推断采样过程

$x_{t-1} = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}} \left[ x_t - \frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}\varepsilon_{\theta}(x_t,t) \right] + \sigma_tI, \ I\sim\mathcal{N}(0,1)$

将$\mu_{\theta}(x_t,t)$代回$L_{t-1}$的表达式中,并将$x_t$写作$x_0$的采样

$L_{t-1} = E_{x_0,\varepsilon}\left[ \frac{1}{2\sigma_t^2} \left\| \sqrt{\frac{\alpha_{t-1}}{\alpha_t}}\frac{\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)}{\sqrt{1-\alpha_t}}(\varepsilon - \varepsilon_{\theta}(x_t,t)) \right\|^2\right] 
= E_{x_0,\varepsilon}\left[ \frac{\alpha_{t-1}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)^2}{2\sigma_t^2\alpha_t(1-\alpha_t)} \left\|\varepsilon - \varepsilon_{\theta}(x_t,t) \right\|^2\right]
=E_{x_0,\varepsilon}\left[ \frac{\alpha_{t-1}\left( 1 - \frac{\alpha_t}{\alpha_{t-1}}\right)^2}{2\sigma_t^2\alpha_t(1-\alpha_t)} \left\|\varepsilon - \varepsilon_{\theta}(\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\varepsilon,t) \right\|^2\right]$

$\left\|\varepsilon - \varepsilon_{\theta}(\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\varepsilon,t) \right\|^2$ 即为训练过程的损失函数

### Codes for DDPM

The code comes from https://github.com/abarankab/DDPM/tree/main/ddpm/diffusion.py
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        img_size,
        img_channels,
        num_classes,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)    # Exponential Moving Average 指数移动平均，更新权重
                                     # weight * EMA_decay + (1 - EMA_decay) * new_weight 
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))                   
        self.register_buffer("alphas", to_torch(alphas))                 
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))  

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))  # 
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
```

$\begin{aligned}
&\text{the register\_buffer} \\
&\text{"betas":} = \beta \\
&\text{"alphas":} = \alpha = 1-beta \\
&\text{"alphas\_cumprod":} = \bar{\alpha} = \prod{\alpha} \\
&\text{"sqrt\_alphas\_cumprod":} = \sqrt{\bar{\alpha}} \\
&\text{"sqrt\_one\_minus\_alphas\_cumprod":} = \sqrt{1 - \bar{\alpha}} \\
&\text{"reciprocal\_sqrt\_alphas":} = \sqrt{\frac{1}{\alpha}} \\
&\text{"remove\_noise\_coeff":} = \frac{\beta}{\sqrt{1 - \bar{\alpha}}} \\
&\text{"sigma":} = \sigma = \sqrt{\beta} \\
\end{aligned}$

```python
    def update_ema(self):  # 判断是否使用EMA方式更新权重
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True): 
        if use_ema:
            return (
                # extract 函数表示从输入的第一个张量tensor中提取步骤t对应的元素再组成(b,1...,1)的tensor，通过broadcastable参与运算
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
```

$\text{remove\_noise 计算} \frac{1}{\sqrt{\alpha_t}}\left( x - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\right) \epsilon_{\theta} \text{获得reverse过程中确定的部分}$

```python
    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True): # 采样过程，从白噪声到图像
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema) 

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x) # 对每步确定的部分附加该步的噪声
                # 和remove_noise合并起来就是原文中Algorithm 2
                # extract 从序列里抽取t步变量，转换成 (batch, 1 * (len(x.shape)-1)) tensor, 维度和x相同， batch值相同的变量
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):  
        #不展示了，和sample作用一样，增加了一个diffusion_sequence用来记录各步的状态

    def perturb_x(self, x, t, noise): #DDPM训练时的diffusion process
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        ) 
```

$\text{perturb\_x表示前向过程} \sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}\epsilon$

```python
    def get_losses(self, x, t, y):  # 完成原文公式(12)
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise) # 第t步的前向结果
        estimated_noise = self.model(perturbed_x, t, y) # 由第t步前向结果预测得到的noise

        # l1，l2目的均使得预测的noise和实际noise最小
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        return loss

    def forward(self, x, y=None): # train过程
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device) # 随机生成步数t,使得训练的网络和step sequence解耦
        return self.get_losses(x, t, y)


def generate_cosine_schedule(T, s=0.008): # beta的生成策略
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2 # approve diffusion的cos策略
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999)) # DDPM标准策略
    
    return np.array(betas)
```

## Score-based SDE

Paper: [Score-Based GEnerative Modeling Through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456)

Score-based SDE 通过将从分布$x(0)~p_0$添加噪声转变为$x(T)~p_T$的马尔可夫扩散过程描述为如下随机微分方程($It\hat{o} \ SDE$)的形式。

$$dx = f(x,t)dt + g(t)dw$$

由上式可以获得对应的反向微分方程(reverse-time SDE),即从$p_T$中恢复$p_0$的分布。

$$dx = [f(x,t)-g^2(t)\triangledown_xlogp_t(x)]dt + g(t)dw$$

要证明上式过程的成立则需要证明Kolmogorov forward equation和Kolmogorov backward equation.

### Kolmogorov forward equation (KFE)

对于给定的随机微分方程

$$dx = f(x,t)dt + g(t)dw$$

,可推出Fokker-Planck方程:

$$\frac{\partial P(x,t)}{\partial t} = -\frac{\partial \left[\mu(x,t)P(x,t) \right]}{\partial x} + \frac{1}{2}\frac{\partial^2 \left[ \sigma^2(x,t)P(x,t) \right]}{\partial x^2}$$

对Fokker-Planck方程的证明，参考《The Fokker-Plank equation》

Diffusion的前向过程可以表示为马尔可夫添加噪声的过程,对于前向过程$X$时刻$t_2$变化到时刻$t_1,t_1>t_2$,值从$x_2$变化到$x_1$。随机变量$x_1,x_2$的联合概率$p(x_1,x_2) = p(x_1|x_2)p(x_2)$.随机变量的$x_1$在$x_2$的后验概率$p(x_1|x_2)$等于从$t_1$到$t_2$时刻的状态转移概率$q_{t_1,t_2}(x_1,x_2)$.

$$p(x_1,x_2) = q_{t_1,t_2}(x_1,x_2)p(x_2)$$

$p(x_1)$的概率可表示为$x_1,x_2$的联合概率对$x_2$的积分

$$p(x_1) = \int q_{t_1,t_2}(x_1,x_2)p(x_2)dx_2$$

对状态转移概率$q_{t_1,t_2}(x_1,x_2),t_1>t_2$,考虑$t_1$到$t_2$作为马尔可夫过程的$1$步状态转移可写作:

$$p(x_{t1})=p(x_{t2})q_{t_1,t_2}(x_1,x_2)$$

这里引入特征函数(Characteristic function),对随机变量$X$的特征函数$\varphi_X(t)$的定义如下：

$$\varphi_X(t) = Ee^{itX} = \int e^{itx}dF_X(x) = \int e^{itx}p(x)dx$$

式中$p(x)$表示随机变量$X$的概率密度函数,$F_X(x)$表示随机变量$X$的分布函数

对$1$步马尔可夫过程状态转移概率其有特征函数：

$$
Ee^{itX_{t1}} = Ee^{itX_{t2}}\varphi_X(q_{t_1,t_2}(x_1,x_2)) \\
\varphi_X(q_{t_1,t_2}(x_1,x_2)) = \frac{Ee^{itX_{t1}}}{Ee^{itX_{t2}}} = Ee^{it(X_{t1}-X_{t1})}
$$

对q_{t_1,t_2}(x_1,x_2)有如下等式成立:

$$q_{t_1,t_2}(x_1,x_2) = \frac{1}{2\pi}\int_{\infty}^{-\infty}e^{-it(x_1-x_2)}Ee^{it(X_{t1}-X_{t2})}dt$$

其证明过程如下:

对特征函数$\varphi_X(t)$有其反演定理如下:

$$F_X(x_2)-F_X(x_1)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{e^{-itx_1}-e^{-itx_2}}{it}\varphi_X(t)dt$$

反演定理的证明:(参考 https://zhihu.com/question/359049277)

第一步:证明狄利克雷积分$\int_0^{\infty}\frac{sint}{t}dt = \frac{\pi}{2}$

$\int_0^{\infty} e^{-th}dh = \int_0^{\infty} -\frac{1}{t} e^{-th}d(-th) = \int_0^{\infty} -\frac{1}{t} -\frac{1}{t} de^{-th} = -\frac{1}{t}\left( e^{-th}|_{h=\infty}-e^{-th}|_{h=0}\right) = -\frac{1}{t}(0-1) = \frac{1}{t}$

狄利克雷积分可表示为

$\int_0^{\infty}\frac{sint}{t}dt = \int_0^{\infty}sint\int_0^{\infty}e^{-th}dhdt = \int_0^{\infty}\int_0^{\infty}e^{-th}sintdtdh$

对其中的$\int_0^{\infty}e^{-th}sintdt$的项使用分步积分法

$\int_0^{\infty}e^{-th}sintdt = -\frac{1}{h} \int^{\infty}_0 sint de^{-th} =  -\frac{1}{h} \left( sinte^{-th} - \int^{\infty}_0 e^{-th}costdt \right)  = -\frac{1}{h} \left( sinte^{-th} + \frac{1}{h} \int^{\infty}_0 costde^{-th} \right) = -\frac{1}{h}sinte^{-th} - \frac{1}{h^2} \left( coste^{-th} + \int^{\infty}_0 e^{-th}sintdt\right)$
 
$\frac{1+h^2}{h^2}\int^{\infty}_0 e^{-th}sintdt = -\frac{1}{h^2}\left( hsinte^{-th} + coste^{th}\right)$

$\int^{\infty}_0\int^{\infty}_0 e^{-th}sintdt = -\frac{1}{1 + h^2}\left( hsinte^{-th} + coste^{th}\right)|^{t=\infty}_{t=0} = -\frac{1}{1 + h^2}(0 - 1) = \frac{1}{1 + h^2}$

$\int_0^{\infty}e^{-th}sintdtdh = \int_0^{\infty}\frac{1}{1 + h^2}dh = arctanh|^{\infty}_0 = \pi/2$

进一步,对积分$\int^{\infty}_0 \frac{sin(\alpha t)}{t}dt$,根据$s=\alpha t, t=\frac{s}{\alpha}$,用s替换t

$\begin{aligned}
& \int^{\infty}_0 \frac{sin(\alpha t)}{t}dt = \int^{\infty}_0 \frac{sin(s)}{s}ds \\
& if \ \alpha>0, \int^{\infty}_0 \frac{sin(s)}{s}ds = \frac{\pi}{2}, if \ \alpha<0, \int^{\infty}_0 \frac{sin(\alpha t)}{t}dt = \int^{\infty}_0 \frac{sin(-s)}{-s}d-s = - \int^{\infty}_0 \frac{sin(s)}{s}ds = -\frac{\pi}{2}
\end{aligned}$

第二步：对于反演定理其右侧

$\begin{aligned}
&\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{e^{-itx_1}-e^{-itx_2}}{it}\varphi_X(t)dt \\
&\text{对}\varphi_X(t)\text{用其变形做}\int e^{itx}dF_X(x)\text{替换} \\
& =\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{e^{-itx_1}-e^{-itx_2}}{it}\int e^{itx}dF_X(x)dt
= \frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\frac{e^{-it(x-x_1)}-e^{-it(x-x_2)}}{it} dtdF_X(x) \\
& \text{将复函数展开为三角函数的形式} \\
& = \frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\frac{( i*sin(t(x-x_1)) + cos(t(x-x_1))) - ( i*sin(t(x-x_2)) + cos(t(x-x_2)))}{it}dtdF_X(t) \\
& \because \frac{cos(t(x-x_n))}{it}\text{是关于t的基函数，其在}(-\infty,\infty)\text{区间上积分等于0} \\
& = \frac{1}{2\pi}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\frac{sin(t(x-x_1))-sin(t(x-x_2))}{t}dtdF_X(x) \\
& \frac{sin(t(x-x_n))}{t}\text{是关于t的偶函数，根据偶函数在}(-\infty,\infty)\text{区间上积分的性质} \\
& = \frac{1}{\pi}\int_{-\infty}^{\infty}\int_{0}^{\infty}\frac{sin(t(x-x_1))-sin(t(x-x_2))}{t}dtdF_X(x) \\
& if \ x>max(x_1,x_2)\ or\ x<min(x_1,x_2),\int_{0}^{\infty}\frac{sin(t(x-x_1))}{t}-\frac{sin(t(x-x_2))}{t}dt = 0 \\
& if \ x_1<x<x_2,\int_{0}^{\infty}\frac{sin(t(x-x_1))}{t}-\frac{sin(t(x-x_2))}{t}dt = \pi \\
& if \ x\in\{x_1,x_2\},\int_{0}^{\infty}\frac{sin(t(x-x_1))}{t}-\frac{sin(t(x-x_2))}{t}dt = \frac{\pi}{2} \\
& = \int_{x_2}^{x_1}dF_X(x) + \int0.5dF_X(x)|_{x=x_1,x=x_2} = F_X(x_2) - F_X(x_1) + 0.5p(x_1)dx + 0.5p(x_2)dx \\
& \because dx \to 0, \therefore p(x)dx \to 0 \\
& = F_X(x_2) - F_X(x_1)
\end{aligned}$

对概率密度$p(x)$有：

$\begin{aligned}
p(x) &= \underset{h\to0}{lim}\frac{F_X(x+h)-F_X(x-h)}{2h}=\underset{h\to0}{lim}\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{e^{-it(x-h)}-e^{-it(x+h)}}{2ith}\varphi_X(t)dt \\
&= \underset{h\to0}{lim}\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{e^{ith}-e^{-ith}}{2ith}e^{-itx}\varphi_X(t)dt = \underset{h\to0}{lim}\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{2isin(th)}{2ith}e^{-itx}\varphi_X(t)dt \\
&= \underset{h\to0}{lim}\frac{1}{2\pi}\int_{-\infty}^{\infty}\frac{sin(th)}{th}e^{-itx}\varphi_X(t)dt = \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{-itx}\varphi_X(t)dt
\end{aligned}$

所以对Diffusion前向过程状态转移概率有:

$$q_{t_1,t_2}(x_1,x_2) = \frac{1}{2\pi}\int_{\infty}^{-\infty}e^{-it(x_1-x_2)}Ee^{it(X_{t1}-X_{t2})}dt$$

这里外侧的$e^{-it(x_{t1}-x_{t2})}$写作$e^{-it(x_1-x_2)}$的原因是由对时刻$t_1,t_2$确定的随机变量$X_{t1},X_{t2}$的状态为$x_1,x_2$.

所以有

$$p(x_1) = \int q_{t_1,t_2}(x_1,x_2)p(x_2)dx_2 = \int \frac{1}{2\pi}\int_{\infty}^{-\infty}e^{-it(x_1-x_2)}Ee^{it(X_{t1}-X_{t2})}dt \ p(x_2)dx_2$$

对$Ee^{it(X_{t1}-X_{t2})}$参考: [Moment generation function](https://en.wikipedia.org/wiki/Moment-generating_function)

$Ee^{it(X_{t1}-X_{t2})} = \sum_{n\ge0}\frac{(it)^n}{n!}E\left( \left< x_{t1} - x_{t2}\right>^n \right)$

$\begin{aligned}
p(x_1) & = \int \frac{1}{2\pi}\int_{\infty}^{-\infty}e^{-it(x_1-x_2)}\sum_{n\ge0}\frac{(it)^n}{n!}E\left( \left< x_{t1} - x_{t2}\right>^n \right)dt \ p(x_2)dx_2 \\
& \because E\left( \left< x_{t1} - x_{t2}\right>^n \right)\text{不包含参数t,对积分进行整理} \\
& = \sum_{n\ge0} \frac{(-1)^n}{n!} \int \frac{1}{2\pi}\int_{\infty}^{-\infty}(-it)^n e^{-it(x_1-x_2)}dtE\left( \left< x_{t1} - x_{t2}\right>^n \right)p(x_2)dx_2 \\
& \because \text{对脉冲函数}\delta_a\text{的特征函数为}e^{ita} \\
& = \sum_{n\ge0} \frac{(-1)^n}{n!} \int \frac{\partial}{\partial x_1}\delta(x_1-x_2) E\left( \left< x_{t1} - x_{t2}\right>^n \right)p(x_2)dx_2 \\
& \text{令}x_1 = x_\tau,x_2=x,t1 = \tau,x_t2=x \\
& = \sum_{n\ge0} \frac{(-1)^n}{n!} \int \frac{\partial}{\partial x_\tau}\delta(x_\tau-x) E\left( \left< x_\tau - x\right>^n \right)p(x)dx \\
& \because \delta(x_\tau-x)dx=1 \\
& = \sum_{n\ge0} \frac{(-1)^n}{n!} \int \frac{\partial}{\partial x_\tau}\delta(x_\tau-x) E\left( \frac{\left< x_\tau - x\right>^n}{\tau} \right)\tau p(x) \\
& = \sum_{n\ge0} \frac{(-1)^n}{n!} \int \frac{\partial}{\partial x_\tau} E\left( \frac{\left< x_\tau - x\right>^n}{\tau} \right) \tau p(x)
\end{aligned}$

由上式可以推出$p_x$对$t$的导数$\dot{p_x}$

$$\dot{p_x} = \underset{\tau\to0}{lim} \sum_{n\ge0} \frac{(-1)^n}{n!} \frac{\partial}{\partial x_\tau} E\left( \frac{\left< x_\tau - x\right>^n}{\tau} \right) \tau p(x) $$

引入Diffusion的前向过程的随机微分方程$dx = f(x,t)dt + g(t)dw$，$E\left( \frac{\left< x_\tau - x\right>^n}{\tau} \right)$, 对随机过程变量$X$其在微小时间内变化量的的期望$E\left< x_{t+dt} - x_t\right> = \mu(X_t,t)dt$,变化量的方差$E\left< x_{t+dt} - x_t\right>^2 = \sigma^2(X_t,t)dt$.保留上式级数中的前两阶。

$$\begin{aligned}
\frac{\partial P(x,t)}{\partial t}  &= -\frac{\partial f(x,t)p(x)}{\partial x} + \frac{1}{2}\frac{\partial^2 g^2(t)p(x)}{\partial x^2} \\
\end{aligned}$$

由$\partial_t P(x,t)$可以得到对应的Probability flow ODE

$\begin{aligned}
-\partial_t p(x_t) &= \partial_x f(x_t)p(x_t) - \frac{1}{2}\partial_x^2g^2(t)p(x_t) \\
&= \partial_x f(x_t)p(x_t) - \partial_x \frac{1}{2} \partial_xg^2(t)p(x_t) - \partial_x \frac{1}{2}g^2(t)\partial_x p(x_t) \\
&=\partial_x f(x_t)p(x_t) - \partial_x \frac{1}{2} \partial_xg^2(t)p(x_t) - \partial_x\frac{1}{2}g^2(t)p(x_t)\frac{1}{p(x_t)}\partial_x p(x_t) \\
&=\partial_x\left[ f(x_t)p(x_t) - \frac{1}{2} \partial_xg^2(t)p(x_t) - \frac{1}{2}g^2(t)p(x_t)\triangledown_{x_t}logp(x_t)\right] \\
&=\partial_x\left[ f(x_t) - \frac{1}{2} \partial_xg^2(t) - \frac{1}{2}g^2(t)\triangledown_{x_t}logp(x_t)\right] p(x_t)
\end{aligned}$

$\partial_xg^2(t) = 0$,根据$It\hat{o} \ SDE$到fokker-plank方程的变化可以令

$\tilde{f}(x_t) = f(x_t) - \frac{1}{2}g^2(t)\triangledown_{x_t}logp(x_t)$

获得相对应的微分方程

$dx = \tilde{f}(x_t) dt = [f(x_t) - \frac{1}{2}g^2(t)\triangledown_{x_t}logp(x_t)]dt$

### Reverse Time Stochastic Differential Equations

Score based SDE 的reverse过程，在前向过程(diffusion process)通过时间$t$和状态$X_t$的条件向后演化概率分布式，在稍后的时间点$s$其随机量$X_s$的概率如何变化。在这里Anderson(1982)引入了两个时刻随机量的联合概率$p(x_s,x_t)$,$p(x_s,x_t)=p(x_s|x_t)p(x_t),t\le s$,讨论概率的变化率对$p(x_s,x_t)$对时间求偏导，作为reverse对其取负，有：
$$-\partial_t p(x_s,x_t) = -\partial_t[p(x_s|x_t)p(x_t)] = \underset{KBE}{\underbrace{-\partial_t p(x_s|x_t)}}p(x_t) - p(x_s|x_t)\underset{KFE}{\underbrace{\partial_t p(x_t)}}$$

对于KFE和KBE过程的证明参考：[FOKKER, PLANK & KOLMOGOROV REVISITED](https://ludwigwinkler.github.io/blog/Kramers)和[Kolmogorov Back Equation: Derivation and Interpretation](https://www.youtube.com/watch?v=wrvHHNCRl7I)

#### Kolmogorov forward equation (KFE) 的证明

对Kolmogorov forward equation的证明即获得$\partial_t p(x_t)$的表达

假设时间$t \le t',t'=t+\tau$,$x,x'$对应$t,t'$时刻的Markov过程随机变量。对$p(x_{t'}')$引入Chapman-Kolmogorov equation有：

$$p(x_{t'}') = \int p(x_{t'}',x_t)dx_t = \int p(x_{t'}'|x_t)p(x_t)dx_t$$

这里Chapman-Kolmogorov equation表示为对nuisance variable(额外变量)$x_t$的简单边缘化(即从联合概率分布求解边缘概率分布)。

后边的过程为了简化，当$x$和$t$的上标一致时，只保留$x$的上标.

引入泰勒展开：$f(x)$在$x=a$处的泰勒展开$T_{f,a}(x)=\sum^\infty_{n=0}\frac{f^{(n)}(a)}{n!}(x-a)^n$,

改写一下$x$为$a+(x-a)$，$f(a+(x-a))$在$x=a$处的泰勒展开与上式相同，其中$x-a\to0$,令$x-a=h,h\to0$,$f(a+h)$在$a$处的泰勒展开$T_{f,a}(h)=\sum^\infty_{n=0}\frac{f^{(n)}(a)}{n!}h^n$.

对$p(x'|x)$引入泰勒展开，

$\begin{aligned}
p(x'|x) &= \int \delta(y_{t'}-x')p(y_{t'}|x) dy_{t'} \\
&= \int \delta(y_{t'}-x + x -x')p(y_{t'}|x) dy_{t'} \\
& \text{函数} \delta(y_{t'}-x + x -x') \text{在} x \text{处展开,} h = y_{t'}-x, a = x -x' \\
&= \sum^\infty_{n=0}\frac{1}{n!}\partial^n_x\delta(x -x') \int(y_{t'}-x)^np(y_{t'}|x) dy_{t'} 
\end{aligned}$

$p(x')$的方程转化为$p(x') = \int \sum^\infty_{n=0}\frac{1}{n!}\partial^n_x\delta(x -x') \int(y_{t'}-x)^np(y_{t'}|x) dy_{t'} p(x)dx$

考虑$It\hat{o} \ SDE$所描述的微分方程$dx = f(x,t)dt + g(t)dw$

$f(x,t)$是$X_t$的漂移项，表示$X_t$的值的变化速率。将$X_t$的变化展开为一个时间很小的过程：

$\int(x'-x)^np(x'|x) dx'$,式中$x'-x$表示$X_t$的变化量，p(x'|x)表示$X_t$由值$x$向值$x'$变化的状态转移概率，$\int(x'-x)p(x'|x) dx'$表示在确定了$X_t$在$t$时刻的状态$x$向下一临近时刻任意状态$x'$变化时变化量的概率平均，要表示$X_t$的值的变化速率，需要对其进行时间上的归一化(除以变化时间$\tau$),有

$f(x,t)=\frac{1}{\tau}\int(x'-x)p(x'|x) dx'$

同理有：

$g^2(t)=\frac{1}{\tau}\int(x'-x)^2p(x'|x) dx'$

在[FOKKER, PLANK & KOLMOGOROV REVISITED](https://ludwigwinkler.github.io/blog/Kramers)中对于该部分的证明有误，可以参考[Kolmogorov Back Equation: Derivation and Interpretation](https://www.youtube.com/watch?v=wrvHHNCRl7I)里的证明。

将$\int(x'-x)^n(x'|x) dx'$描述为某种增量的$k$阶矩$M^{(n)}(x),f(x,t)\tau=M^{(1)}(x),g^2(x,t)\tau=M^{(2)}(x)$,将其代入$p(x')$的表达式。

$$p(x') = \sum^\infty_{n=0}\frac{1}{n!} \int \partial^n_x\delta(x -x') M^{(n)}(x) p(x)dx$$

引入分部积分：
$$\int^\infty_{-\infty} \delta^{(n)}(x)f(x)dx = \sum^{n-1}_k[(-1)^k\delta^{(k)}(x)f^{(n-k)}(x)] |^\infty_{-\infty}+(-1)^n\int^\infty_{-\infty}\delta(x)f^{(n)}(x)dx$$

上式中$f^{(n)}(\pm \infty)=f(\pm \infty)=0$,概率密度函数在$\pm \infty$处等于0，在$\pm \infty$处的$n$阶导也为0.

有：$\sum^{n-1}_k[(-1)^k\delta^{(k)}(x)f^{(n-k)}(x)] |^\infty_{-\infty}=0,\ \int^\infty_{-\infty} \delta^{(n)}(x)f(x)dx = (-1)^n\int^\infty_{-\infty}\delta(x)f^{(n)}(x)dx$

$$\begin{aligned}
p(x') &= \sum^\infty_{n=0}\frac{1}{n!} \int \partial^n_x\delta(x -x') M^{(n)}(x) p(x)dx \\
&= \sum^\infty_{n=0}\frac{1}{n!} (-1)^n \int\delta(x -x') \partial^n_x [M^{(n)}(x) p(x)]dx \\
&\text{保留前两阶} \\
&= p(x) - \partial_x [M^{(1)}(x) p(x)] + \frac{1}{2} \partial^2_x [M^{(2)}(x) p(x)] \\
&= p(x) - \partial_x [f(x,t)\tau p(x)] + \frac{1}{2} \partial^2_x [g^2(t)\tau p(x)] \\
&\text{移项} \\
p(x')-p(x) &= - \tau \partial_x [f(x,t) p(x)] + \tau \frac{1}{2} \partial^2_x [g^2(t) p(x)] \\
\frac{p(x')-p(x)}{\tau} &= -\partial_x [f(x,t) p(x)] + \frac{1}{2} \partial^2_x [g^2(t) p(x)] \\
\partial_t p(x) &= -\partial_x [f(x,t) p(x)] + \frac{1}{2} \partial^2_x [g^2(t) p(x)]
\end{aligned}$$

获得了Kolmogorov forward equation (KFE) 

$$\partial_t p(x) = -\partial_x [f(x,t) p(x)] + \frac{1}{2} \partial^2_x [g^2(t)p(x)]$$

#### Kolmogorov backward equation (KBE) 的证明

对Kolmogorov backward equation的证明即获得$\partial_t p(x_t|x_{t'}')$的表达

假设时间$t'\le t'' \le t,t'' = t'+\tau$,$x'',x',x$对应$t'',t',t$时刻的Markov过程随机变量。对$p(x_t|x_{t'}')$引入Chapman-Kolmogorov equation有：

$$p(x_t|x_{t'}') = \int p(x_t|x_{t''}'') p(x_{t''}''|x_{t'}')dx_{t''}''$$

Chapman-Kolmogorov equation 的结果证明如下：

$\begin{aligned}
\int p(x_{t'}',x_{t''}'',x_t) dx_{t''}'' & = p(x_{t'}',x_t) \\
\int p(x_{t'}')p(x_{t''}''|x_{t'}')p(x_t|x_{t''}'') dx_{t''}'' & = p(x_{t'}',x_t) \\
\int p(x_{t''}''|x_{t'}')p(x_t|x_{t''}'') dx_{t''}'' & = \frac{p(x_{t'}',x_t)}{p(x_{t'}')} \\
\int p(x_t|x_{t''}'')p(x_{t''}''|x_{t'}') dx_{t''}'' & = p(x_t|x_{t'}')
\end{aligned}$

Chapman-Kolmogorov equation 中对$x_{t''}''$的积分也可以理解为由$t'$时刻的确定的状态$x'$变换到$t''$的任意状态$x''$,再由任意状态$x''$变换到$t$时刻的确定的状态$x_t$的过程。

同样为了简化，之后当$x$和$t$的上标一致时，只保留$x$的上标.

对$p(x''|x')$引入泰勒展开，

$\begin{aligned}
p(x''|x') &= \int \delta(y_{t''}-x'')p(y_{t''}|x') dy_{t''} \\
&= \int \delta(y_{t''}-x' + x' -x'')p(y_{t''}|x') dy_{t''} \\
& \text{函数} \delta(y_{t''}-x' + x' -x'') \text{在} x' \text{处展开,} h = y_{t''}-x', a = x' -x'' \\
&= \sum^\infty_{n=0}\frac{1}{n!}\partial^n_{x'}\delta(x' -x'') \int(y_{t''}-x')^np(y_{t''}|x') dy_{t''} \\
&= \sum^\infty_{n=0}\frac{1}{n!}\partial^n_{x'}\delta(x' -x'') M^{(n)}(x') \tau
\end{aligned}$

$\begin{aligned}
p(x|x') &= \int p(x|x'')p(x''|x')dx'' \\
&= \int p(x|x'') \sum^\infty_{n=0}\frac{1}{n!}\partial^n_{x'}\delta(x' -x'') M^{(n)}(x')\tau dx'' \\
&\text{因为}t'\text{时的矩}M^{(n)}(x')\text{和}t''\text{时刻的随机变量}x''\text{相互独立，可以从积分中提取出来} \\
&= \sum^\infty_{n=0}\frac{1}{n!} M^{(n)}(x')\tau \int p(x|x'')\partial^n_{x'}\delta(x' -x'')dx'' \\
&\text{引入分部积分} \\
&= \sum^\infty_{n=0}\frac{1}{n!} M^{(n)}(x')\tau \partial^n_{x'} \int p(x|x'')\delta(x' -x'')dx'' \\
&\text{对}\delta()\text{的积分，可以视为对变量}x''\text{取}x'\text{的值} \\
&= \sum^\infty_{n=0}\frac{1}{n!} M^{(n)}(x')\tau \partial^n_{x'} p(x|x_{t''}') \\
&\text{保留前两阶} \\
&= p(x|x_{t''}') + M^{(1)}(x')\partial_{x'} p(x|x_{t''}')\tau + \frac{1}{2} M^{(2)}(x')\partial^2_{x'} p(x|x_{t''}')\tau 
\end{aligned}$

$$\begin{aligned}
p(x|x') &= p(x|x_{t''}') + M^{(1)}(x')\partial_{x'} p(x|x_{t''}')\tau + \frac{1}{2} M^{(2)}(x')\partial^2_{x'} p(x|x_{t''}')\tau \\
&\text{移项} \\
p(x|x') - p(x|x_{t''}') &= M^{(1)}(x')\partial_{x'} p(x|x_{t''}')\tau + \frac{1}{2} M^{(2)}(x')\partial^2_{x'} p(x|x_{t''}')\tau \\
&\text{因为时间}t''\ge t'(x'\text{对应时间}t'),\text{因为时间}t''\text{和}t'\text{相隔时间短,近似的}p(x|x_{t''}')\approx p(x|x') \\
- \frac{p(x|x_{t''}') - p(x|x')}{\tau} &= M^{(1)}(x')\partial_{x'} p(x|x') + \frac{1}{2} M^{(2)}(x')\partial^2_{x'} p(x|x') \\
-\partial_t p(x|x') &= f(x', t) \partial_{x'} p(x|x') + \frac{1}{2} g^2(t) \partial^2_{x'} p(x|x')
\end{aligned}$$

获得了Kolmogorov backward equation (KBE) 

$$-\partial_t p(x|x') = f(x', t) \partial_{x'} p(x|x') + \frac{1}{2} g^2(t) \partial^2_{x'} p(x|x')$$

#### Reverse Time Stochastic Differential Equations 的证明

将KFE和KBE代入$-\partial_t p(x_s,x_t)$

参考[REVERSE TIME STOCHASTIC DIFFERENTIAL EQUATIONGS [ FOR GENERATIVE MODELLING]](https://ludwigwinkler.github.io/blog/ReverseTimeAnderson)

$\begin{aligned}
-\partial_t p(x_s,x_t) &= -\partial_t p(x_s|x_t)p(x_t) - p(x_s|x_t)\partial_t p(x_t) \\
&= \left( f(x_t) \partial_{x_t} p(x_s|x_t) + \frac{1}{2} g^2(t) \partial^2_{x_t} p(x_s|x_t)\right)p(x_t) + p(x_s|x_t)\left( \partial_{x_t} [f(x_t) p(x_t)] - \frac{1}{2} \partial^2_{x_t} [g^2(t)p(x_t)]\right)
\end{aligned}$

对上式中的未知的项$\partial_{x_t} p(x_s|x_t), \partial_{x_t} [f(x_t) p(x_t)], \partial^2_{x_t} [g^2(t) p(x_t)]$进行求解。

$\partial_{x_t} p(x_s|x_t) = \partial_{x_t} \left[ \frac{p(x_s,x_t)}{p(x_t)}\right] = \frac{\partial_{x_t} p(x_s,x_t)p(x_t)-p(x_s,x_t)\partial_{x_t}p(x_t)}{p^2(x_t)}$

$\partial_{x_t} [f(x_t) p(x_t)] = \partial_{x_t}f(x_t) p(x_t) + f(x_t) \partial_{x_t}p(x_t)$

$\partial^2_{x_t} [g^2(t) p(x_t)] = g^2(t) \partial^2_{x_t} p(x_t)$

将上面几个式子代入$-\partial_t p(x_s,x_t)$的表达式中

$\begin{aligned}
-\partial_t p(x_s,x_t) &= f(x_t)\frac{\partial_{x_t} p(x_s,x_t)p(x_t)-p(x_s,x_t)\partial_{x_t}p(x_t)}{p^{\cancel{2}}(x_t)}\cancel{p(x_t)} + \frac{1}{2} g^2(t) \partial^2_{x_t} p(x_s|x_t) p(x_t) + p(x_s|x_t)\partial_{x_t}f(x_t) p(x_t) + f(x_t) p(x_s|x_t)\partial_{x_t}p(x_t) - \frac{1}{2} p(x_s|x_t) \partial^2_{x_t} [g^2(t)p(x_t)] \\
&= f(x_t)\partial_{x_t} p(x_s,x_t) - \cancel{f(x_t) p(x_s|x_t) \partial_{x_t}p(x_t)} + p(x_s|x_t)p(x_t)\partial_{x_t}f(x_t) + \cancel{f(x_t) p(x_s|x_t)\partial_{x_t}p(x_t)} + \frac{1}{2} g^2(t) \partial^2_{x_t} p(x_s|x_t) p(x_t) - \frac{1}{2} p(x_s|x_t) \partial^2_{x_t} [g^2(t)p(x_t)] \\
&= \underset{\text{乘法求导法则}}{\underbrace{f(x_t)\partial_{x_t} p(x_s,x_t) + p(x_s,x_t) \partial_{x_t}f(x_t)}} + \frac{1}{2} g^2(t) \left[\partial^2_{x_t} p(x_s|x_t) p(x_t) - p(x_s|x_t)\partial^2_{x_t} p(x_t)\right] \\
&= \partial_{x_t} [f(x_t) p(x_s,x_t)] + \frac{1}{2} g^2(t) \left[\partial^2_{x_t} p(x_s|x_t) p(x_t) - p(x_s|x_t)\partial^2_{x_t} p(x_t)\right]
\end{aligned}$

其中对$\partial^2_{x_t} p(x_s|x_t) p(x_t) - p(x_s|x_t)\partial^2_{x_t} p(x_t)$引入二阶微分的展开

$\begin{aligned}
\partial^2_{x_t}p(x_s,x_t) &= \partial^2_{x_t}[p(x_s|x_t)p(x_t)] = \partial^2_{x_t}p(x_s|x_t)p(x_t) + 2\partial_{x_t}p(x_s|x_t)\partial_{x_t}p(x_t) + p(x_s|x_t)\partial^2_{x_t}p(x_t) \\
& = \left[\partial^2_{x_t} p(x_s|x_t) p(x_t) - p(x_s|x_t)\partial^2_{x_t} p(x_t)\right] + 2\partial_{x_t}p(x_s|x_t)\partial_{x_t}p(x_t) + 2p(x_s|x_t)\partial^2_{x_t}p(x_t) \\
\end{aligned}$

$\begin{aligned}
\partial^2_{x_t} p(x_s|x_t) p(x_t) - p(x_s|x_t)\partial^2_{x_t} p(x_t) &= \partial^2_{x_t}p(x_s,x_t) - 2\partial_{x_t}p(x_s|x_t)\partial_{x_t}p(x_t) - 2p(x_s|x_t)\partial^2_{x_t}p(x_t) \\
&= \partial^2_{x_t}p(x_s,x_t) - 2 \left[ \underset{\text{乘法求导法则}}{\underbrace{\partial_{x_t}p(x_s|x_t)\partial_{x_t}p(x_t) + p(x_s|x_t)\partial^2_{x_t}p(x_t)}}\right] \\
&= \partial^2_{x_t}p(x_s,x_t) - 2\partial_{x_t}\left[p(x_s|x_t)\partial_{x_t}p(x_t) \right]
\end{aligned}$

上式代入$-\partial_t p(x_s,x_t)$

$\begin{aligned}
-\partial_t p(x_s,x_t) &= \partial_{x_t} [f(x_t) p(x_s,x_t)] + \frac{1}{2} g^2(t)\partial^2_{x_t}p(x_s,x_t) - g^2(t) \partial_{x_t}\left[p(x_s|x_t)\partial_{x_t}p(x_t) \right] \\
&= \partial_{x_t} \left[ f(x_t) p(x_s,x_t) - g^2(t) p(x_s|x_t)\partial_{x_t}p(x_t)\right] + \frac{1}{2} g^2(t)\partial^2_{x_t}p(x_s,x_t) \\
&= \partial_{x_t} p(x_s,x_t) \left[ f(x_t) - g^2(t) \frac{1}{p(x_t)}\partial_{x_t}p(x_t)\right] + \frac{1}{2} g^2(t)\partial^2_{x_t}p(x_s,x_t) \\
&= \partial_{x_t} p(x_s,x_t) \left[ f(x_t) - g^2(t) \partial_{x_t} logp(x_t) \right] + \frac{1}{2} g^2(t)\partial^2_{x_t}p(x_s,x_t)
\end{aligned}$

对$x_s$做积分获得$-\partial_t p(x_s,x_t)$的边缘分布$-\partial_t p(x_t)$

$$-\partial_t p(x_t) = \partial_{x_t} \left[ f(x_t) - g^2(t) \partial_{x_t} logp(x_t) \right] p(x_t) + \frac{1}{2} g^2(t)\partial^2_{x_t} p(x_t)$$

再参考前向过程SDE和KFE的形式，可以获得reverse-time SDE

$$dX_t = \left[ f(x_t) - g^2(t) \partial_{x_t} logp(x_t) \right] dt + g(t)dw$$

### VP-SDE 和 VE-SDE

对于SMLD([Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600)) 即往已知信号中不断地添加噪声的过程，文中描述为VE-SDE(Variance Exploding 噪声爆炸)，其前向过程数学描述为

$$x_i = x_0 + \sigma_iZ_i$$

把上式写成差分的形式有

$$\begin{aligned}
x_{i+1} &= x_0 + \sigma_{i+1}Z_{i+1} \\
x_i &= x_0 + \sigma_iZ_i \\
x_{i+1} &= x_i + \sigma_{i+1}Z_{i+1} - \sigma_iZ_i =  x_i + \sqrt{\sigma^2_{i+1}-\sigma^2_i}Z_{i} \\
dx &= \sqrt{\sigma^2(t+\delta t)-\sigma^2(t)}Z_{i} = \sqrt{\frac{d[\sigma^2]}{dt}dt}Z_{i} \\
dX_t &= \sqrt{\frac{d[\sigma^2]}{dt}}w_i
\end{aligned}$$

对DDPM([Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)) 即diffusion过程中已知信号和添加噪声的的总尺度不变，文中描述为VP-SDE(Variance Preserving), DDPM的前向过程描述为

$$x_i = \sqrt{1-\beta_i}x_{i-1}+\beta_iz_{i-1}$$

假设前向过程有$N$步，对每步的噪声尺度有$\{ \beta_i = \beta(t)\}$ 即将噪声尺度变化写作关于尺度随时间变化的函数

$$\begin{aligned}
x(t+\triangle t) &= \sqrt{1-\beta(t+\triangle t)\triangle t}x(t)+\sqrt{\beta(t+\triangle t)\triangle t}z(t) \\ 
&\sqrt{1-\beta(t+\triangle t)\triangle t}\text{对}\triangle t\text{作泰勒展开,} \triangle t \to 0\\
x(t+\triangle t) &= (1 - \frac{1}{2}\beta(t+\triangle t)\triangle t + o(\triangle t))x(t)+\sqrt{\beta(t+\triangle t)\triangle t}z(t) \\
&\approx x(t) - \frac{1}{2}\beta(t+\triangle t)\triangle t x(t) + \sqrt{\beta(t+\triangle t)\triangle t}z(t) \\
&\approx x(t) - \frac{1}{2}\beta(t) \triangle t x(t) \sqrt{\beta(t)\triangle t}z(t)
\end{aligned}$$

$$dx = -\frac{1}{2}\beta(t)xdt+\sqrt{\beta(t)}dw$$

### Codes for Score-based SDE

The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/run_lib.py
```python
def train(config, workdir): 
    """Runs the training pipeline.
    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    ...
    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    # Setup SDE  选择SDE前向过程(diffusion process,ancestral sampling)的类型给sampling和loss计算提供对应类型的计算函数
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous  # 控制前向过程的计算方式
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    # losses.get_step_fn返回loss_fn的函数实例，这里用了闭包的写法
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,reduce_mean=reduce_mean, continuous=continuous,likelihood_weighting=likelihood_weighting)
    ...
    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
        # 这里同样采用了闭包的写法，返回的是内部函数sampling_fn的实例
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    num_train_steps = config.training.n_iters
    ...
    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        # Execute one training step
        """train_step_fn这里才是真正的训练"""
        loss = train_step_fn(state, batch) # 计算该步的训练loss
        ...
        # Generate and save samples
        if config.training.snapshot_sampling:
            ...
            sample, n = sampling_fn(score_model) # 调用sampling_fn在驯良过程中对模型进行采样保存
            ...
```
The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py
```python
def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.
    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    # continuous时离散的采样步被归一化到[0,1]区间内
    def loss_fn(model, batch):
        """Compute the loss function.
        Args:
        model: A score model.
        batch: A mini-batch of training data.
        Returns:
        loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps # 生成[0,1]区间的采样时间
        z = torch.randn_like(batch) # 生成batch维噪声，对应batch个输入信号,噪声为标准正态分布，噪声尺度为之后的std
        mean, std = sde.marginal_prob(batch, t) # 根据SDE过程推导的p(x(t)|x(0))的分布的mean和std
        perturbed_data = mean + std[:, None, None, None] * z # continuous的diffusion过程
        score = score_fn(perturbed_data, t) #通过score_fn计算，而非其他loss通过model_fn计算

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        # 这里存在问题，
        # 对VPSDE,score_fn的输出时-noise_pred/ancestral_sampling_variance, 
        #         对应的loss = -noise_pred/ancestral_sampling_variance*distribuate_variance
        #         loss+z=0
        # 对subVPSDE,score_fn的输出时-noise_pred/distribuate_variance, 
        #            对应的loss = -noise_pred/distribuate_variance*distribuate_variance
        #            loss+z=0
        # 但对VESDE,score_fn输出=model_fn，则对应的loss是什么？        
        loss = torch.mean(losses)
        return loss
    return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."
    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device) # labels随机的时间步数，有btach维不同，之后的也一样
        sigmas = smld_sigma_array.to(batch.device)[labels] #VE noise 对应labels的噪声尺度
        noise = torch.randn_like(batch) * sigmas[:, None, None, None] #VE noise
        perturbed_data = noise + batch # ancestral sampling获得diffusion前向后模糊的数据
        score = model_fn(perturbed_data, labels) # 模型输出的score function结果
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target) # 这一时间步的loss
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2 #这里对不同时间步的loss进行加权
        loss = torch.mean(losses)
        return loss
    return loss_fn
```
$\begin{aligned}
&\text{smld loss 来自《Generative Modeling by Estimating Gradients of the Data Distribution》的4.2 Learning NCSNs via score matching} \\
&\text{对一个VE-SDE过程其第i步的数据分布有}q_{\sigma}(\tilde{x}|x)=\mathcal{N}(\tilde{x}|x,\sigma^2I)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\tilde{x}-x)^2}{2\sigma^2}} \\
&\text{所以损失函数}l(\Theta;\sigma)=E_{q_{\sigma}(\tilde{x}|x)p_{data}(x)}\left\| s_{\theta}(\tilde{x})-\triangledown_{\tilde{x}}logq_{\sigma}(\tilde{x}|x)\right\|^2_2 = E_{q_{\sigma}(\tilde{x}|x)p_{data}(x)}\left\| s_{\theta}(\tilde{x}) - \frac{\tilde{x}-x}{\sigma^2}\right\|^2_2 \\
&\mathcal{L}(\Theta;\{\sigma_i\}^L_{i=1}) = \frac{1}{L}\sum^L_{i=1}\lambda(\sigma_i)l(\Theta;\sigma_i),\lambda(\sigma_i)=\sigma^2_i
\end{aligned}$
```python
def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device) #和smld_loss中一样，batch维采样时间步
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device) # ancestral sampling的信号系数
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device) # ancestral sampling的噪声系数
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None] * noise # ancestral sampling获得diffusion前向后模糊的数据
        score = model_fn(perturbed_data, labels) # 输出预测的噪声
        losses = torch.square(score - noise) # 计算预测噪声和实际噪声的l2 loss
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss #ddpm的loss已经在之前介绍过了
    return loss_fn

def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
    Returns:
        A one-step function for training or evaluation.
    """
    # 根据continuous和SDE类型选择不同的loss计算方式
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.
        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.
        Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.
        Returns:
            loss: The average loss value of this state.
        """
        """diffusion过程的前向，模型的前向计算，loss计算，以及反向更新均在这里完成，所以称这里是真正的训练"""
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch) # 包含diffusion前向，模型前向和loss计算
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step']) # 参数更新
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad(): # 对数滑动平均更新模型
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())
        return loss
    return step_fn
```
The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/models/utils.py
```python
def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    Returns:
        A score function.
    """
    model_fn = get_model_fn(model, train=train)
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
        # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999 # 对continous，通过将(0,1)区间转化为相应的非整数的步数输入网络
                score = model_fn(x, labels) # 实际的网络输出
                std = sde.marginal_prob(torch.zeros_like(x), t)[1] # 根据SDE过程推导的p(x(t)|x(0))的分布的std
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()] # ancestral sampling对应的std

        score = -score / std[:, None, None, None] # 这里的目标是使得loss预测中model_fn输出等于noise
        return score
    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1] # VESDE对应的p(x(t)|x(0))的分布的std
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                # 在train过程中未使用，目的是将连续时间转换成近似步数
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
        score = model_fn(x, labels)
        return score
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    return score_fn
```
The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py
```python
class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""
    ...
    def discretize(self, x, t): # 基类的离散化前向过程
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)
        Returns:
            f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t) #从前向SDE获得
        f = drift * dt 
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        # 计算f,G随时间的累积量作为随机变量x的变化量dx=f(x,t)dt+g(t)\sqrt(t)
        return f, G

    def reverse(self, score_fn, probability_flow=False): # reverse过程返回reverse过程的drift，和随机系数
        """Create the reverse-time SDE/ODE.
        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t): # 
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t) # 根据前向过程的SDE方程获得f(x,t),g(t)
                score = score_fn(x, t) # 获得logp(t)对x的导数即score function
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t): # 离散化的reverse采样
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t) # 根据离散化的前向SDE方程获得f(x,t),g(t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()
```
$\begin{aligned}
&\text{reverse过程的SDE} \\
&\text{以}\partial_tp(x_s,x_t)\text{作为reverse过程}dx=\left[f(x,t)-g^2(t)\triangledown_xlogp_t(x)\right]dt+g(t)dw \\
&\text{以}\partial_tp(x_s)\text{的负作为reverse过程}dx=\left[f(x,t)-\frac{1}{2}g^2(t)\triangledown_xlogp_t(x)\right]dt+g(t)dw
\end{aligned}$
```python
class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
```
$\begin{aligned}
&\text{beta\_0:} \bar{\beta}_{min},\text{beta\_1:} \bar{\beta}_{max},\text{discrete\_betas:}\beta(t) = \bar{\beta}_{min}+t(\bar{\beta}_{max}-\bar{\beta}_{min}) \\
&\text{alphas:} \alpha_t = 1 - \beta(t),\text{alphas\_cumprod:} \bar{\alpha}_t = \prod\alpha_t,\text{sqrt\_alphas\_cumprod:} \sqrt{\bar{\alpha}_t},\text{sqrt\_1m\_alphas\_cumprod:}\sqrt{1-\bar{\alpha}_t}
\end{aligned}$
```python
    @property
    def T(self):
        return 1

    def sde(self, x, t): # VP-SDE 前向随机微分方程
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion
```
$\text{VP SDE方程:} dx = -\frac{1}{2}\beta(t)dt + \sqrt{\beta(t)}dw$
```python    
    def marginal_prob(self, x, t): # x(t)对x(0)的概率分布参数
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
```
$\begin{aligned}
&p_{0t}(x(t)|x(0)) = \mathcal{N}\left( x(t);e^{-\frac{1}{4}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-\frac{1}{2}t\bar{\beta}_{min}}x(0),1-e^{-\frac{1}{2}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-t\bar{\beta}_{min}}\right) \\
&\text{证明:} \\
&\text{对于均值，引入随机微分方程的解，参考https://en.wikipedia.org/wiki/Stochastic\_differential\_equation} \\
&\text{对型如:}dX_t = (a(t)X_t+c(t))dt + (b(t)X_t+d(t))dW_t\text{的随机微分方程,} \\
&\text{有其解的形式} X_t = \Phi_{t,t_0}\left( X_{t_0} + \int^t_{t_0}\Phi^{-1}_{s,t_0}(c(s)-b(s)d(s))ds + \int^t_{t_0}\Phi^{-1}_{s,t_0}d(s)dW_s\right) \\
&\Phi_{t,t_0} = exp\left(\int^t_{t_0}\left(a(s)-\frac{b^2(s)}{2}\right)ds+\int^t_{t_0}b(s)dW_s\right) \\
&\text{代入VP-SDE方程:}dx = -\frac{1}{2}\beta(t)xdt+\sqrt{\beta(t)}dw,\text{有:}a(t)=-\frac{1}{2}\beta(t),b(t)=c(t)=0,d(t)=\sqrt{\beta(t)} \\
&\Phi_{t,t_0} = exp\left(\int^t_{t_0}-\frac{1}{2}\beta(s)ds\right)=exp\left(-\frac{1}{4}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-\frac{1}{2}t\bar{\beta}_{min}\right) \\
&E(X_t) = E\left[\Phi_{t,t_0}\left( X_{t_0}+ \int^t_{t_0}\Phi^{-1}_{s,t_0}d(s)dW_s\right)\right],\text{因为}W_s\text{是维纳过程,所以有}E(W_s)=0,E(X_t)=\Phi_{t,t_0}X_0= e^{-\frac{1}{4}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-\frac{1}{2}t\bar{\beta}_{min}}x(0)\\
&\text{对于方差，参考原文引用《Applied Stochastic Differential Equations》式(5.51)} \\
&\frac{dP}{dt} = E[f(x,t)(x-m)^T]+E[(x-m)^Tf^T(x,t)]+E[L(x,t)QL^T(x,t)] \\
&\text{代入VP-SDE方程}where \ m(t)=E(X_t),P=E[(x(t)-m(t))(x(t)-m(t))^T]=\Sigma(X_t),f(x)=-\frac{1}{2}\beta(t)x,L(x,t)=\sqrt{\beta(t)},Q=var(W)=I \\
&\text{有}E[f(x,t)(x-m)]=E[-\frac{1}{2}\beta(t)x(x-m)]=-\frac{1}{2}\beta(t)\left[E(x^2)-E^2(x)\right]=-\frac{1}{2}\beta(t)\Sigma(X_t),\text{代回式(5.51)} \\
&\frac{d\Sigma(X_t)}{dt} = -\beta(t)\Sigma(X_t) + \beta(t) = \beta(t)(I-\Sigma(X_t)),\text{将此式看作}\Sigma(X_t)\text{的常微分方程} \\
&\text{求解}\frac{1}{I-\Sigma(X_t)}d\Sigma(X_t) = \beta(t)dt,-dln(I-\Sigma(X_t))=\beta(t)dt,-ln\left(\frac{I-\Sigma(X_t)}{I-\Sigma(X_0)}\right)=\int\beta(t)dt,I-\Sigma(X_t) = e^{-\int\beta(t)dt}(I-\Sigma(X_0)),\Sigma(X_t)=I-e^{-\int\beta(t)dt}(I-\Sigma(X_0)) \\
&\text{因为VP-SDE,}\Sigma(X_0)=0,\Sigma(X_t)=I-e^{-\int\beta(t)dt} = I-e^{2\int-\frac{1}{2}\beta(t)dt}\\
&p_{0t}(x(t)|x(0))\text{得证} \\
&\text{log\_mean\_coeff}=e^{\int-\frac{1}{2}\beta(t)dt}=e^{-\frac{1}{4}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-\frac{1}{2}t\bar{\beta}_{min}} \\
&\text{mean}=e^{log\_mean\_coeff}x(0),\text{std}=\sqrt{1-e^{2*log\_mean\_coeff}}
\end{aligned}$
```python
    def prior_sampling(self, shape): #推断过程的初始采样
        return torch.randn(*shape)

    def prior_logp(self, z): # 推断过程VP_SDE初始正态分布的对数概率密度函数,方差为1
        shape = z.shape
        N = np.prod(shape[1:]) # np.prod连乘，这里N=dim1*dim2*dim3表示batch中每个对象包含多少个元素
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps
```
$\begin{aligned}
&\text{总概率密度函数}\prod_{k,i,j}^{k\in C,i\in H, j\in W}p(x_{i,j,k}) = \prod_{k,i,j}^{k\in C,i\in H, j\in W} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\
&\sigma = 1,x-\mu = z \\
&\text{对数概率密度}log(\prod_{k,i,j}^{k\in C,i\in H, j\in W} \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}) = \sum_{k,i,j} -\frac{1}{2} log(2\pi\sigma) - \sum_{k,i,j}\frac{(x_{k,i,j}-\mu)^2}{2\sigma^2} = -\frac{N}{2} log(2\pi) - \sum_{k,i,j}\frac{z^2_{k,i,j}}{2}, N = C*H*W
\end{aligned}$
```python
    def discretize(self, x, t): # 根据ancestral sampling离散化的前向过程
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long() #将连续时间离散化成步数
        beta = self.discrete_betas.to(x.device)[timestep] 
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G
```
$dx_t = \sqrt{1-\beta(t)}x_tdt + \sqrt{\beta(t)}dW_t, f=\sqrt{1-\beta(t)}, G=\sqrt{\beta(t)},\beta(t)=\bar{\beta}_{min}+t(\bar{\beta}_{max}-\bar{\beta}_{min})$
```python
class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t): # sub-VP SDE 前向随机微分方程
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion
```
$\text{sub-VP SDE方程:} dx = -\frac{1}{2}\beta(t)dt + \sqrt{\beta(t)(1-e^{-2\int_0^t\beta(s)ds})}dw,\beta(t)=\bar{\beta}_{min}+t(\bar{\beta}_{max}-\bar{\beta}_{min})$
```python
    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std
```
$\begin{aligned}
&\text{和VP SDE采用同样的方法}E(X_t)=\Phi_{t,t_0}X_0 = e^{\int-\frac{1}{2}\beta(t)dt}x(0)= e^{-\frac{1}{4}t^2(\bar{\beta}_{max}-\bar{\beta}_{min})-\frac{1}{2}t\bar{\beta}_{min}}x(0) \\
&\text{同样参考原文引用列出方差方程}\frac{d\Sigma(X_t)}{dt} = -\beta(t)\Sigma(X_t) + \beta(t)(1-e^{-2\int_0^t\beta(s)ds}) = \beta(t)\left(1-e^{-2\int_0^t\beta(s)ds}-\Sigma(X_t)\right) \\
&\text{这里的证明利用微分方程的解，将上式写成非齐次微分方程的形式} \\
&\frac{d\Sigma(X_t)}{dt} + \beta(t)\Sigma(X_t) = \beta(t)(1-e^{-2\int_0^t\beta(s)ds}) \\
&\text{直接套用非齐次微分方程的通解形式} \\
&\Sigma(X_t) = Ce^{-\int\beta(t)dt}+e^{-\int\beta(t)dt}\int\beta(t)(1-e^{-2\int_0^t\beta(s)ds})e^{\int\beta(t)dt}dt \text{\ 这个方程是可以积分的，不需要用到什么求解的奇计淫巧} \\
&\text{先不管齐次部分，保留非齐次部分，其中}\beta(t)e^{\int\beta(t)dt} = de^{\int\beta(t)dt} \\
&e^{-\int\beta(t)dt}\int\beta(t)(1-e^{-2\int_0^t\beta(s)ds})e^{\int\beta(t)dt}dt = e^{-\int\beta(t)dt}\int1-e^{-2\int_0^t\beta(s)ds}de^{\int\beta(t)dt} = e^{-\int\beta(t)dt}\left(e^{\int\beta(t)dt}+e^{-\int\beta(t)dt}\right) = I+e^{-2\int\beta(t)dt} \\
&\Sigma(X_t) = Ce^{-\int\beta(t)dt} + I + e^{-2\int\beta(t)dt},\Sigma(X_0) = Ce^{-\int\beta(t)dt}|_{t=0}+2I = C + 2I,C = \Sigma(X_0) - 2I \\
&\Sigma(X_t) = I + e^{-2\int\beta(t)dt} + e^{-\int\beta(t)dt}(\Sigma(X_0) - 2I) \\
&\text{因为}\Sigma(X_0)=0,\Sigma(X_t) = I + e^{-2\int\beta(t)dt} + 2I*e^{-\int\beta(t)dt} = \left(1-e^{-\int\beta(t)dt}\right)^2
\end{aligned}$
```python
    def prior_sampling(self, shape): #推断过程的初始采样
        return torch.randn(*shape)

    def prior_logp(self, z): # 推断过程subVP-SDE初始正态分布的对数概率密度函数
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.  


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.
        Args:
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
```
$\text{discrete\_sigmas:}\sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^{\frac{i-1}{N-1}}$
```python
    @property
    def T(self):
        return 1

    def sde(self, x, t): # VE-SDE 前向随机微分方程
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion
```
$\text{VE SDE方程}dx = \sqrt{\frac{d\sigma^2(t)}{dt}}dw=\sqrt{2\sigma(t)\frac{\sigma(t)}{dt}}dw=\sqrt{2\sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t\sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^tlog\frac{\sigma_{max}}{\sigma_{min}}}dw=\sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t\sqrt{2log\frac{\sigma_{max}}{\sigma_{min}}}dw$
```python
    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
```
$\begin{aligned}
&\text{从SMLD的马尔可夫过程,ancestral sampling可以得到VE SDE的}p_{\sigma_i}(x|x_0) \\
&x_i=x_{i-1}+\sqrt{\sigma^2_i-\sigma^2_{i-1}}z_{i-1},\text{\ 递推下去，则有}E(x_i) = x_0,\Sigma(x_i)=\sum_{i=0}^i\sigma^2_i-\sigma^2_{i-1} = \sigma^2_i - \sigma^2_0 \\
&\sigma(t)=\sigma(\frac{i}{N}) = \sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^{\frac{i-1}{N-1}}= \sigma_{min}\left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t,t=\frac{i-1}{N-1}\in[0,1]
\end{aligned}$
```python
    def prior_sampling(self, shape):  #推断过程的初始采样
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z): # 推断过程VE-SDE初始正态分布的对数概率密度函数,其方差为sigma_max
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t): # VE SDE的ancestral sampling离散化前向过程
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                    self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G
```
$dx = \sqrt{\sigma_i^2-\sigma_{i-1}^2}z_{i-1}$

The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
```python
...
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t) # self.rsde继承自Predictor，表示reverse过程
    z = torch.randn_like(x)
    x_mean = x - f # 单步的drift
    x = x_mean + G[:, None, None, None] * z # 单步的随机化
    return x, x_mean
...
def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.
    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t): # 单步的reverse过程的drift
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True) # 获得模型输出的score function实例,logp(x)对x的导数
        rsde = sde.reverse(score_fn, probability_flow=True) # 获得reverse过程的微分方程
        return rsde.sde(x, t)[0] #返回reverse过程的drift

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.
        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device) # 初始时间对应的采样
            else:
                x = z

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t #时间向量
            drift = drift_fn(model, x, vec_t) #SDE 的reverse过程的
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                        rtol=rtol, atol=atol, method=method)
        # scipy.intergrate.solve_ivp根据输入的微分方程ode_func依照(sde.T, eps)总时长和步长，进行采样仿真，计算每步的X的值
        nfe = solution.nfev # 总点数
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32) # 采样结果
        # Denoising is equivalent to running one predictor step without adding noise
        if denoise:
            x = denoise_update_fn(model, x) 
        x = inverse_scaler(x) # 从归一化数据恢复原值
        return x, nfe

    return ode_sampler    
```
The code comes from https://github.com/yang-song/score_sde_pytorch/blob/main/likelihood.py
```python
def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
            x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.
    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
    Returns:
        A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True) # 获取score function
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True) # 获取reverse过程的SDE
        return rsde.sde(x, t)[0] # reverse过程的drift方程

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)
        # 按照 drift_fn(model, xx, tt)定义get_div_fn的参数fn
        # 再将变量(x, t, noise)输给该闭包计算其内部函数div_fn的输出

    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
        model: A score model.
        data: A PyTorch tensor.

        Returns:
        bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
        z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
        nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with torch.no_grad():
        shape = data.shape
        if hutchinson_type == 'Gaussian':
            epsilon = torch.randn_like(data)
        elif hutchinson_type == 'Rademacher':
            epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def ode_func(t, x):
            sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
            vec_t = torch.ones(sample.shape[0], device=sample.device) * t
            drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
            logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
            return np.concatenate([drift, logp_grad], axis=0)

        init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
        # scipy.integrate.solve_ivp对ode_func按eps时长,sde.T步长进行仿真运算
        nfe = solution.nfev
        zp = solution.y[:, -1] #对数导数score function
        z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
        delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
        prior_logp = sde.prior_logp(z) #根据z获得z的对数概率分布
        bpd = -(prior_logp + delta_logp) / np.log(2) #？？？
        N = np.prod(shape[1:])
        bpd = bpd / N
        # A hack to convert log-likelihoods to bits/dim
        offset = 7. - inverse_scaler(-1.)
        bpd = bpd + offset
        return bpd, z, nfe

    return likelihood_fn
```