# Diffusion Accelerate

因为Diffusion模型采用逐渐的去噪声过程，使得虽然和其他生成式网络相比虽然精确度领先，但是往往需要很长的步骤在推断效率上不如其他的生成式网络，所以产生了对推断过程加速的需求。

## Denoising Diffusion Implicit Models (DDIM)

Paper:[Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502)

我自己认为DDPM对$X_{t-1}$的估计的主要影响因素是$X_t$,而DDIM的对$X_{t-1}$的估计的主要影响因素是$X_0$

直接根据VP SDE的ancestral sampling过程

$$
X_{t-1} = \sqrt{\alpha_{t-1}}X_0 + \sqrt{1-\alpha_{t-1}}\epsilon
$$

则有

$$
q_{\phi}(X_{t-1}|X_t,X_0) = \mathcal{N}\left(\sqrt{\alpha_{t-1}}X_0 + \sqrt{1-\alpha_{t-1}-\sigma^2}\frac{X_t-\sqrt{\alpha_t}X_0}{\sqrt{1-\alpha_t}}，\sigma^2I\right)
$$

$\begin{aligned}
&\text{其中:}\frac{X_t-\sqrt{\alpha_t}X_0}{\sqrt{1-\alpha_t}}=\epsilon_{\theta}^t\text{表示根据}X_t\text{估计的扩散噪声},\sigma\text{表示过程附加的随机量} \\
&\text{由此可得}E[q_{\phi}(X_{t-1}|X_t,X_0)] = \sqrt{\alpha_{t-1}}X_0, D[q_{\phi}(X_{t-1}|X_t,X_0)] = 1-\alpha_{t-1} \\
&\text{对随机项}\sigma,\sigma=0,\text{则上式为DDIM的形式,若}\sigma=\sqrt{(1-\alpha_{t-1})/(1-\alpha_t)}\sqrt{1-\alpha_t/\alpha_{t-1}}\text{则上式为DDPM的形式}
\end{aligned}$

$\sigma=0$时DDIM的表示形式有

$$\begin{aligned}
X_{t-1} &= \sqrt{\alpha_{t-1}}X_0 + \sqrt{1-\alpha_{t-1}}\frac{X_t-\sqrt{\alpha_t}X_0}{\sqrt{1-\alpha_t}} \\
&= \sqrt{\alpha_{t-1}}X_0 + \sqrt{1-\alpha_{t-1}}\epsilon_{\theta}^t \\
&\text{用}X_t\text{表示}X_0 \\
&= \sqrt{\alpha_{t-1}}\left(\frac{X_t-\sqrt{1-\alpha_{t-1}}\epsilon_{\theta}^t}{\sqrt{\alpha_t}}\right)+ \sqrt{1-\alpha_{t-1}}\epsilon_{\theta}^t \\
\frac{X_{t-1}}{\sqrt{\alpha_{t-1}}} &= \frac{X_t}{\sqrt{\alpha_t}} + \left(\frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{\alpha_{t-1}}}-\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)\epsilon_{\theta}^t \\
&\text{改变采样间隔} \\
\frac{X_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} &= \frac{X_t}{\sqrt{\alpha_t}} + \left(\frac{\sqrt{1-\alpha_{t-\Delta t}}}{\sqrt{\alpha_{t-\Delta t}}}-\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)\epsilon_{\theta}^t \\
&\text{将}\frac{X_t}{\sqrt{\alpha_t}}\text{移到等式左边，对等式右边乘除}\left(\frac{\sqrt{1-\alpha_{t-\Delta t}}}{\sqrt{\alpha_{t-\Delta t}}}+\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right) \\
\frac{X_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} - \frac{X_t}{\sqrt{\alpha_t}} & = \left(\left(\frac{\sqrt{1-\alpha_{t-\Delta t}}}{\sqrt{\alpha_{t-\Delta t}}}\right)^2-\left(\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)^2\right)\Bigg / \left(\frac{\sqrt{1-\alpha_{t-\Delta t}}}{\sqrt{\alpha_{t-\Delta t}}}+\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)\epsilon_{\theta}^t \\
&\text{因为}\frac{\sqrt{1-\alpha_{t-\Delta t}}}{\sqrt{\alpha_{t-\Delta t}}}\approx\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} \\
\frac{X_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} - \frac{X_t}{\sqrt{\alpha_t}} & = \left(\frac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}-\frac{1-\alpha_t}{\alpha_t}\right)\Bigg / \left(2\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)\epsilon_{\theta}^t \\
\frac{X_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} &= \frac{X_t}{\sqrt{\alpha_t}} + \frac{1}{2}\left(\frac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}-\frac{1-\alpha_t}{\alpha_t}\right)\frac{\sqrt{\alpha_t}}{\sqrt{1-\alpha_t}}\epsilon_{\theta}^t
\end{aligned}$$

上式即为DDIM的递推形式

## DPM-Solver

Paper:[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/pdf/2206.00927)

DPM-solver 表示Markov过程其前向过程状态转移方程如下：

$$q_{0t}(x_t|x_0) = \mathcal{N}(x_t|\alpha_tx_0,\sigma_t^2I)$$

等价于如下随机微分方程

$$dx_t=f(t)x_tdt+g(t)dW_t,\ where \ f(x)=\frac{dlog\alpha_t}{dt},g^2(t)=\frac{d\sigma^2_t}{dt}-2\frac{dlog\alpha_t}{dt}\sigma^2_t$$

证明：

对SDE方程$dx_t=f(t)x_tdt+g(t)dW_t$积分

$\begin{aligned}
x_t &= x_0 + \int_0^tf(s)x_sds + \int_0^tg(s)dW_t = x_0 + \int_0^tf(s)x_sds + \sum_{i=1}^N g(t_i)(W_{t_i}-W_{t_{i-1}}) \\ 
&求期望 \\
E(x_t) &= E(x_0) + \int_0^tf(s)E(x_s)ds + \sum_{i=1}^N g(t_i)E(W_{t_i}-W_{t_{i-1}}) \\
&W_t是维纳过程,W_{t_i}-W{t_{i-1}}\sim\mathcal{N}(0,\Delta t) \\
E(x_t) &= E(x_0) + \int_0^tf(s)E(x_s)ds \\
&两边对t求导 \\
\frac{dE(x_t)}{dt} &= f(t)E(x_t) \\
E(x_t) &= Ce^{\int_0^tf(s)ds}=\alpha_tx_0,E(x_0)=x_0,C=x_0 \\
\int_0^tf(s)ds &= log\alpha_t,f(t) = \frac{dlog\alpha_t}{dt}
\end{aligned}$

参考《Variational Diffusion Models》Appendix A.1

对由随机变量$z_s$得到的$z_t,0 \le s \le t$

$$\begin{aligned}
q(z_t|z_s) = \mathcal{N}(\alpha_{t|s}z_s,\sigma_{t|s}^2I),\alpha_{t|s} = \alpha_t/\alpha_s,\sigma^2_{t|s} = \sigma_t^2-\alpha_{t|s}^2\sigma_s^2
\end{aligned}$$

由$q(x_t|x_s) = \mathcal{N}(\alpha_{t|s}x_s,\sigma_{t|s}^2I)$，可得$x_t = \alpha_{t|s}x_s + \sigma_{t|s}W$

$dx = x_t - x_s = (\alpha_{t|s}-1)x_s\Delta t  + \sigma_{t|s}, (\alpha_{t|s}-1)\Delta t = f(t)\Delta t = dlog\alpha_t$

$\begin{aligned}
\sigma^2_{t|s} &= Var(dx_t) = E[(dx_t)^2] - E^2(dx_t) = E[(E(dx_t)+g(t)dW_t)^2]-E^2(dx_t) \\
&= 2E(dx_t)E(g(t))E((W_t-W_s)) + E^2(g^2(t))E((W_t-W_s)^2) \\
\sigma^2_{t|s} &= g^2(t)\Delta t = \sigma^2_t - (dlog\alpha_t +1)^2 \sigma^2_s = \sigma^2_t - \sigma^2_s - 2dlog\alpha_t\sigma^2_s - (dlog\alpha_t)^2\sigma^2_s \\
&= \Delta \sigma^2_t - 2dlog\alpha_t\sigma^2_s - \mathcal{O}
\end{aligned}$

$g^2(t) = \underset{\Delta t \to 0}{lim} \frac{\Delta \sigma^2_t}{\Delta t} - 2 \frac{dlog\alpha_t}{\Delta t}\sigma^2_s = \frac{d\sigma_t^2}{dt} - 2\frac{dlog\alpha_t}{dt}\sigma^2_t$

根据Score-based SDE的reverse SDE

$$dx_t = \left[f(t)x_t-g^2(t)\triangledown_{x_t}log(p(x_t))\right]dt + g(t)dW_t$$

等价于

$$dx_t = \left[f(t)x_t+\frac{g^2(t)}{\sigma_t}\epsilon_{\theta}(x_t,t)\right]dt + g(t)dW_t$$

证明，Teweedie Estimator $p(x|\theta)\sim \mathcal{N}(\theta,\sigma^2)$描述过程基于未知参数$\theta$的观测数据$x$符合正态分布$\mathcal{N}(\theta,\sigma^2)$

$\begin{aligned}
E(\theta|x)&=\int_{-\infty}^{\infty}\theta p(\theta|x)d\theta = \int_{-\infty}^{\infty}\theta \frac{p(x|\theta)p(\theta)}{p(x)}d\theta = \frac{\int_{-\infty}^{\infty}\theta p(x|\theta)p(\theta)d\theta}{p(x)} = \frac{\int_{-\infty}^{\infty}(\theta-x)p(x|\theta)p(\theta)d\theta+\int_{-\infty}^{\infty}xp(x|\theta)p(\theta)d\theta}{p(x)} = \frac{\int_{-\infty}^{\infty}(\theta-x)p(x|\theta)p(\theta)d\theta}{p(x)} + x \\
&=\frac{\int_{-\infty}^{\infty}\sigma^2\frac{\theta-x}{\sigma^2}\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\theta)^2}{\sigma^2}}p(\theta)}{p(x)} +x = \frac{\sigma^2\int_{-\infty}^{\infty}p(\theta)\frac{dp(x|\theta)}{dx}d\theta}{p(x)} + x = \frac{\sigma^2\frac{d}{dx}p(x)}{p(x)}+x = \frac{\sigma^2}{p(x)}\frac{dp(x)}{dx} + x = \sigma^2\frac{dlogp(x)}{dx} + x
\end{aligned}$

对$x_t = \mathcal{N}(\alpha_tx_0,\sigma_tI)=\alpha_tx_0+\sigma_t\epsilon_t$有$E(x_t)=\alpha_tx_0=x_t+\sigma^2\triangledown_{x}logp(x_t)$

$\frac{x_t+\sigma^2\triangledown_{x}logp(x_t)}{\alpha_t}= \frac{x_t-\sigma_t\epsilon_t}{\alpha_t}, \triangledown_{x}logp(x_t) = \frac{\epsilon_t}{\alpha_t}$

等价已证

$dx_t = \left[f(t)x_t+\frac{g^2(t)}{\sigma_t}\epsilon_{\theta}(x_t,t)\right]dt + g(t)dW_t$ 对应的常微分方程为

$$\frac{dx_t}{dt}=f(x)x_t+\frac{g^2(t)}{\sigma_t}\epsilon_{\theta}(x_t,t)$$

同样有Probability flow ODE

$$\frac{dx_t}{dt}=f(x)x_t+\frac{g^2(t)}{2\sigma_t}\epsilon_{\theta}(x_t,t)$$

代入信噪比(SNR)$\lambda_t = log\frac{\alpha_t}{\sigma_t}$

$
g^2(t) = \frac{d\sigma_t^2}{dt} - 2\frac{dlog\alpha_t}{dt}\sigma^2_t = 2\sigma_t\frac{d\sigma_t}{dt} - 2\sigma^2_t\frac{dlog\alpha_t}{dt} = 2\sigma^2_t\frac{1}{\sigma_t}\frac{d\sigma_t}{dt} - 2\sigma^2_t\frac{dlog\alpha_t}{dt} =  2\sigma^2_t\left(\frac{dlog\sigma_t}{dt}-\frac{dlog\alpha_t}{dt}\right) = -2\sigma^2_t\frac{dlog\alpha_t-dlog\sigma_t}{dt} = -2\sigma^2_t\frac{d\lambda_t}{dt}
$

求Probability flow常微分方程的解有：

$$\begin{aligned}
x_t &= e^{\int_s^tf(\tau)d\tau}x_s + \int_s^t\left(e^{\int_{\tau}^tf(\tau)d\tau}\frac{g^2(\tau)}{2\sigma_{\tau}}\epsilon_{\theta}(x_{\tau},\tau)\right)d\tau \\
&= e^{log\alpha_t-log\alpha_s}x_s - \int_s^t\left(e^{log\alpha_t-log\alpha_{\tau}}\sigma_{\tau}\frac{d\lambda_{\tau}}{d\tau}\epsilon_{\theta}(x_{\tau},\tau)\right)d\tau \\
&= \frac{\alpha_t}{\alpha_s}x_s - \int^t_s \left(\frac{\alpha_t}{\alpha_{\tau}}\sigma_{\tau}\frac{d\lambda_{\tau}}{d\tau}\epsilon_{\theta}(x_{\tau},\tau)\right)d\tau \\
&= \frac{\alpha_t}{\alpha_s}x_s - \alpha_t\int^t_s \frac{\sigma_{\tau}}{\alpha_{\tau}}\epsilon_{\theta}(x_{\tau},\tau)d\lambda_{\tau} \\
&= \frac{\alpha_t}{\alpha_s}x_s - \alpha_t\int^t_s e^{-\lambda_{\tau}}\epsilon_{\theta}(x_{\tau},\tau)d\lambda_{\tau} \\
&= \frac{\alpha_t}{\alpha_s}x_s - \alpha_t\int^{\lambda_t}_{\lambda_s} e^{-\lambda} \epsilon_{\theta}(x_{\lambda},\lambda)d\lambda
\end{aligned}$$

DPM-solver的ODE表达式为

$$
x_t = \frac{\alpha_t}{\alpha_s}x_s - \alpha_t\int^{\lambda_t}_{\lambda_s} e^{-\lambda} \epsilon_{\theta}(x_{\lambda},\lambda)d\lambda
$$

对上式做泰勒展开，可以获得DPM-solver的各阶表达式

$$
x_t = \frac{\alpha_t}{\alpha_s}x_s - \alpha_t \sum^{k-1}_{n=0}\epsilon^{(n)}_{\theta}(x_s,\lambda_s)\int_{\lambda_s}^{\lambda_t}\frac{e^{-\lambda}(\lambda-\lambda_s)^n}{n!}d\lambda + \mathcal{O}(h^{k+1})
$$

### 一阶DPM-solver和DDIM等价

$k=1$时,DPM-Solver-1

$\begin{aligned}
x_t &= \frac{\alpha_t}{\alpha_s}x_s - \alpha_t \epsilon_{\theta}(x_s,\lambda_s)\int_{\lambda_s}^{\lambda_t}e^{-\lambda}d\lambda 
=\frac{\alpha_t}{\alpha_s}x_s - \alpha_t \epsilon_{\theta}(x_s,\lambda_s) (-e^{-\lambda})|_{\lambda_s}^{\lambda_t}
=\frac{\alpha_t}{\alpha_s}x_s - \alpha_t \epsilon_{\theta}(x_s,\lambda_s) (e^{-\lambda_s}-e^{-\lambda_t}) \\
&= \frac{\alpha_t}{\alpha_s}x_s - \alpha_t e^{-\lambda_t} \epsilon_{\theta}(x_s,\lambda_s) (e^{\lambda_t-\lambda_s}-1) =  \frac{\alpha_t}{\alpha_s}x_s - \alpha_t \frac{\sigma_t}{\alpha_t} \epsilon_{\theta}(x_s,\lambda_s) (e^{\lambda_t-\lambda_s}-1) = \frac{\alpha_t}{\alpha_s}x_s - \sigma_t \epsilon_{\theta}(x_s,\lambda_s) (e^{\lambda_t-\lambda_s}-1)
\end{aligned}$

对DDIM有

$\begin{aligned}
\frac{X_s}{\sqrt{\alpha_s}} &= \frac{X_t}{\sqrt{\alpha_t}} + \left(\frac{\sqrt{1-\alpha_s}}{\sqrt{\alpha_s}}-\frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\right)\epsilon_{\theta}^t
\end{aligned}$

其前向为$x_t = \sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon_t$,其$\alpha_t = \sqrt{\alpha_t}, \sigma_t = \sqrt{1-\alpha_t}$

上式可化简为：

$X_t = \frac{\alpha_t}{\alpha_s}X_s - \alpha_t(e^{-\lambda_t}-e^{-\lambda_t})\epsilon_{\theta}^t$

等于对DPM-Solver-1推导的中间项，所以一阶DPM-solver和DDIM等价