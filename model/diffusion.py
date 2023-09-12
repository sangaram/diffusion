import torch
import torchvision
import matplotlib.image as mimage
import cv2
from torch import nn
from PIL import Image
from typing import Union, List, Tuple, Optional
import math
from .unet import UNet


class GaussianDiffusionModel(nn.Module):
    """The denoising diffusion model.
    
    Parameters
    ----------
    betas : torch.Tensor
        The standard deviations of the diffusion time steps
    model_mean_type: str
        Indicate what is predicted during the denoising process of the image.
        For model_mean_type=xprev, we predict x_{t-1}
        For model_mean_type=xstart, we predict x0
        For model_mean_type=eps, we prodict epsilon
    model_var_type: str
        Indicate the type of variances we use for the denoising process.
        In this implementation the variances are allways fixed. Even if their can be learnt,
        this choice has been made because as explained in the paper, the training is unstable
        with learnt variances.
        For model_var_type=fixedsmall, the variances of the backward process are used.
        For model_var_type=fixedlarde, the variance of the forward process posterions are used.
    loss_type: str
        Indicate the type of loss used for training.
        For loss_type=mse we use the MSE Loss
        For loss_type=kl we use the KL-Divergence
    channels: int
        Indicate the base number of channels used in the convolutions inside residual blocks
    out_channels: int
        Indicate the number of channels of the output image
    ch_mult: Union[List[int], Tuple[int]]
        Gives the multiplication factor of the channels of the residual blocks at each level of the UNet.
    num_res_blocks: int
        The number of residual blocks
    att_levels: Union[List[int], Tuple[int]]
        The levels where an attention block is added
    num_groups: int
        The number of groups used by the group normalization layers inside the UNet.
        Must be diviser of channels
    resample_with_conv: bool
        Whether to upsample or downsample the images with convolution layers
    p: float
        The dropout probability
    """
    def __init__(
        self,
        betas:torch.Tensor,
        model_mean_type:str,
        model_var_type:str,
        loss_type:str,
        channels:int,
        out_channels:int,
        ch_mult:Union[List[int], Tuple[int]],
        num_res_blocks:int,
        att_levels:Union[List[int], Tuple[int]],
        num_groups:int,
        resample_with_conv:Optional[bool]=True,
        p:Optional[float]=.0
    ):
        super().__init__()
        self.betas = betas
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        assert (betas > 0).all() and (betas <= 1).all(), "Error: all betas must lay between 0 and 1."
        self.num_timesteps = betas.shape[0]
        alphas = 1 - betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1]), self.alphas_cumprod[:-1]), dim=0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod) ** 0.5
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = (1. / self.alphas_cumprod) ** 0.5
        self.sqrt_recipm1_alphas_cumprod = (1. / self.alphas_cumprod - 1) ** 0.5

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((self.posterior_variance[1].reshape(1), self.posterior_variance[1:]), dim=0)
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)

        self.denoiser = UNet(
            channels,
            out_channels,
            ch_mult,
            num_res_blocks,
            att_levels,
            num_groups,
            resample_with_conv,
            p
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.betas = self.betas.to(*args, **kwargs)
        self.alphas_cumprod = self.alphas_cumprod.to(*args, **kwargs)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(*args, **kwargs)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(*args, **kwargs)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(*args, **kwargs)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(*args, **kwargs)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(*args, **kwargs)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(*args, **kwargs)
        self.posterior_variance = self.posterior_variance.to(*args, **kwargs)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(*args, **kwargs)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(*args, **kwargs)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(*args, **kwargs)

        return self

    def cpu(self):
        super().cpu()
        self.betas = self.betas.cpu()
        self.alphas_cumprod = self.alphas_cumprod.cpu()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.cpu()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cpu()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.cpu()
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.cpu()
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.cpu()
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.cpu()
        self.posterior_variance = self.posterior_variance.cpu()
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.cpu()
        self.posterior_mean_coef1 = self.posterior_mean_coef1.cpu()
        self.posterior_mean_coef2 = self.posterior_mean_coef2.cpu()

        return self

    def gpu(self):
        super().gpu()
        self.betas = self.betas.cpu()
        self.alphas_cumprod = self.alphas_cumprod.gpu()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.gpu()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.gpu()
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.gpu()
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.gpu()
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.gpu()
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.gpu()
        self.posterior_variance = self.posterior_variance.gpu()
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.gpu()
        self.posterior_mean_coef1 = self.posterior_mean_coef1.gpu()
        self.posterior_mean_coef2 = self.posterior_mean_coef2.gpu()

        return self

    
    def extract(self, x:torch.Tensor, t:torch.Tensor, shape:Tuple[int]) -> torch.Tensor:
        """Extracts the values from a given tensor at indices given by a list.
            Args
            ----
            x: torch.Tensor
                The input tensor
            t: torch.Tensor
                The indices
            shape: Tuple[int]
                The shape of the output
            
            Returns
            -------
            A tensor with shape `shape`with values of `x` at indices specified by `t`
        """

        B, = t.shape
        assert B == shape[0], f"Error: t must have the same length as the batch size. Got {B} and {x.shape[0]}"
        out = x[t]

        return out.reshape(B, *([1]*len(shape[1:])))

    def q_mean_variance(self, x0:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Gets the mean and the variance of the diffusion process at the given given time steps.
            Args
            ----
            x0: torch.Tensor
                The initial images
            t: torch.Tensor
                The time steps

            Returns
            -------
            A tensor with the means and the variances
        """

        mean = self.extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = self.extract(1. - self.alphas_cumprod, t, x0.shape)
        log_variance = self.extract(self.log_one_minus_alphas_cumprod, t, x0.shape)
        return mean, variance, log_variance

    def q_sample(self, xstart:torch.Tensor, t:torch.Tensor, noise:Optional[torch.Tensor]=None) -> torch.Tensor:
        """Samples the resulting image at the given time step from the distribution of the diffusion process
            Args
            ----
            xstart: torch.Tensor
                The input image
            t: torch.Tensor
                The time steps
            noise: Optional[torch.Tensor]
                The noise use to generate the image
            
            Returns
            -------
            The images of the specified time steps
        """

        if noise is None:
            noise = torch.empty_like(xstart).normal_()

        return (
            self.extract(self.sqrt_alphas_cumprod, t, xstart.shape) * xstart +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, xstart.shape) * noise
        )

    def q_posterior_mean_variance(self, xstart:torch.Tensor, xt:torch.Tensor, t:torch.Tensor) -> Tuple[torch.Tensor]:
        """Gets the mean and the variance of the diffusion process posterior at the given time steps.
            Args
            ----
            xstart: torch.Tensor
                The initial images
            t: torch.Tensor
                The time steps

            Returns
            -------
            A tensor with the means and the variances
        """

        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, xt.shape) * xstart +
            self.extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )

        posterior_variance = self.extract(self.posterior_variance, t, xt.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, xt.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_xstart_from_xprev(self, xt:torch.Tensor, t:torch.Tensor, xprev:torch.Tensor) -> torch.Tensor:
        """Predicts x_{t-1} from x_t and x0
        Args
        ----
        xt: torch.Tensor
            The image at time step `t`
        t: torch.Tensor
            The time step
        xprev: torch.Tensor
            The value of x0

        Returns
        -------
        The value of x_{t-1}
        """

        return (  # (xprev - coef2*x_t) / coef1
            self.extract(1. / self.posterior_mean_coef1, t, xt.shape) * xprev -
            self.extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt.shape) * xt
        )


    def predict_xstart_from_eps(self, xt:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        """Predicts x_{t-1} from x_t and the noise
        Args
        ----
        xt: torch.Tensor
            The image at time step `t`
        t: torch.Tensor
            The time step
        eps: torch.Tensor
            The value of the noise

        Returns
        -------
        The value of x_{t-1}
        """

        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt -
            self.extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * eps
        )

    def p_mean_variance(self, x:torch.Tensor, t:torch.Tensor, clip_denoised: bool, return_pred_xstart: bool) -> Tuple[torch.Tensor]:
        """Gets the mean and the variance of the denoising process at the given time step.
            Args
            ----
            x: torch.Tensor
                The value of x_t
            t: torch.Tensor
                The time step

            Returns
            -------
            A tensor with the means and the variances
        """

        B, C, H, W = x.shape
        denoiser_out = self.denoiser(x, t)
        assert tuple(denoiser_out.shape) == (B, C, H, W)

        if self.model_var_type == 'fixedlarge':
            model_variance = self.betas
            model_log_variance = torch.log(torch.cat((self.posterior_variance[1].reshape(1), self.betas[1:])))
        elif self.model_var_type == 'fixedsmall':
            model_variance = self.posterior_variance
            model_log_variance = self.posterior_log_variance_clipped
        else:
            raise NotImplementedError(self.model_var_type)

        model_variance = self.extract(model_variance, t, x.shape) * torch.ones_like(x)
        model_log_variance = self.extract(model_log_variance, t, x.shape) * torch.ones_like(x)

        # Mean parameterization
        _maybe_clip = lambda x_: (torch.clip(x_, -1., 1.) if clip_denoised else x_)
        if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
            pred_xstart = _maybe_clip(self.predict_xstart_from_xprev(x, t, denoiser_out))
            model_mean = denoiser_out
        elif self.model_mean_type == 'xstart':  # the model predicts x_0
            pred_xstart = _maybe_clip(denoiser_out)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        elif self.model_mean_type == 'eps':  # the model predicts epsilon
            pred_xstart = _maybe_clip(self.predict_xstart_from_eps(x, t, denoiser_out))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        if return_pred_xstart:
              return model_mean, model_variance, model_log_variance, pred_xstart
        else:
              return model_mean, model_variance, model_log_variance

    def p_sample(self, x:torch.Tensor, t:torch.Tensor, clip_denoised:bool, return_pred_xstart:bool) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """Samples x_{t-1} from x_t in the distribution of the denoising process
            Args
            ----
            x: torch.Tensor
                The value of x_t
            t: torch.Tensor
                The time step
            
            Returns
            -------
            The the value of x_{t-1}
        """

        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(x, t, clip_denoised, return_pred_xstart=True)
        noise = torch.empty_like(x).normal_()
        nonzero_mask = (t != 0).reshape(-1, *([1]*(len(x.shape)-1))) # shape = (B, 1, 1, ..., 1)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, sample_shape:Tuple[int]) -> torch.Tensor:
        """Loops over the whole denoising process starting from a random noise sample from standard normal distribution.
            Args
            ----
            sample_shape: Tuple[int]
                The shape of the image. Make sure it contains the batch dimension
            
            Returns
            -------
            The generated image
        """

        device = self.betas.device
        sample = torch.empty(*sample_shape, device=device).normal_() # Initialize sample with noise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((sample_shape[0],), i, device=device)
            sample = self.p_sample(sample, t, clip_denoised=True, return_pred_xstart=False)

        return sample

    @torch.no_grad()
    def generate(self, shape:Tuple[int]) -> Image:
        """Generate an new image with the given shape.
            Args
            ----
            sample: Tuple[int]
                The shape of the image.
            
            Returns
            -------
            The generated image
        """
        assert len(shape) >= 3 and len(shape) <= 4, f"Error: Expecting the length of shape to be either 3 or 4"
        if len(shape) == 3:
            shape = (1,) + shape

        img = self.p_sample_loop(shape)
        img = torchvision.transforms.functional.to_pil_image(img[0])
        return img
    
    @torch.no_grad()
    def generation_evolution(self, shape:Tuple[int], filename:str) -> torch.Tensor:
        """Generate an new image with the given shape and the video of the generation process.
            Args
            ----
            sample: Tuple[int]
                The shape of the image.
            filename: str
                The name of the video file.
            Returns
            -------
            The generated image
        """
        assert len(shape) == 3, f"Error: Expecting shape of format (C,H,W)"
        device = self.betas.device
        frame_height = shape[1]
        frame_width = shape[2]
        frame_rate = self.num_timesteps / 30
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        ext = filename.split('.')[-1]
        assert ext == 'mp4', f"Error: Expecting mp4 extension but got {ext}"
        out = cv2.VideoWriter(filename, codec, frame_rate, (frame_width, frame_height))
        shape = (1,) + shape
        img = torch.empty(*shape, device=device).normal_()
        frame = mimage.pil_to_array(torchvision.transforms.functional.to_pil_image(img[0]))
        out.write(frame)
        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i], device=device)
            img = self.p_sample(img, t, clip_denoised=True, return_pred_xstart=False)
            frame = mimage.pil_to_array(torchvision.transforms.functional.to_pil_image(img[0]))
            out.write(frame)
        
        out.release()
        
        return img[0].detach()

    @staticmethod
    def normal_kl(mean1:torch.Tensor, logvar1:torch.Tensor, mean2:torch.Tensor, logvar2:torch.Tensor) -> torch.Tensor:
        """Computes the KL-Divergence of two normal distributions
            Args
            ----
            mean1: torch.Tensor
                The mean of the first distribution
            logvar1: torch.Tensor
                The log-variance of the first distribution
            mean2: torch.Tensor
                The mean of the second distribution
            logvar1: torch.Tensor
                The log-variance of the second distribution
            
            Returns
            -------
            The KL-Divergence
        """
        return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + ((mean1-mean2)**2) * torch.exp(-logvar2))

    @staticmethod
    def approx_standard_normal_cdf(x:torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def discretized_gaussian_log_likelihood(x:torch.Tensor, means:torch.Tensor, log_scales:torch.Tensor) -> torch.Tensor:
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = GaussianDiffusionModel.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
        cdf_min = GaussianDiffusionModel.approx_standard_normal_cdf(min_in)
        c = torch.tensor(1e-12)
        log_cdf_plus = torch.log(torch.maximum(cdf_plus, c))
        log_one_minus_cdf_min = torch.log(torch.maximum(1. - cdf_min, c))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                 torch.log(torch.maximum(cdf_delta, c))))
        assert log_probs.shape == x.shape
        return log_probs

    def variational_bound_terms(self, xstart:torch.Tensor, xt:torch.Tensor, t:torch.Tensor, clip_denoised:bool, return_pred_xstart:bool) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        """Computes the variational bound terms (L_t in the paper)
            Args
            ----
            xstart: torch.Tensor
                The value of x0
            xt: torch.Tensor
                The value of x_t
            t: torch.Tensor
                The time step `t`
            clip_denoised: bool
                Whether to clip the denoised image
            
            Returns
            -------
            The value of L_t
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(xstart, xt, t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
          xt, t, clip_denoised, return_pred_xstart=True)
        kl = self.normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = torch.mean(kl, dim=tuple(range(1, len(kl.shape)))) / math.log(2.)
        assert kl.shape == t.shape

        decoder_nll = -self.discretized_gaussian_log_likelihood(
          xstart, means=model_mean, log_scales=0.5 * model_log_variance)

        decoder_nll = torch.mean(decoder_nll, dim=tuple(range(1, len(decoder_nll.shape)))) / math.log(2.)

        output = torch.where(t == 0, decoder_nll, kl)
        return (output, pred_xstart) if return_pred_xstart else output

    def training_losses(self, xstart:torch.Tensor, t:torch.Tensor, noise:Optional[torch.Tensor]=None) -> torch.Tensor:
        if noise is None:
            noise = torch.empty_like(xstart).normal_()

        xt = self.q_sample(xstart, t, noise)

        if self.loss_type == 'kl':
            losses = self.variational_bound_terms(xstart, xt, t, clip_denoised=False, return_pred_xstart=False)
        elif self.loss_type == 'mse':
            if self.model_mean_type == 'xprev':
                target = self.q_posterior_mean_variance(xstart, xt, t)[0]
            elif self.model_mean_type == 'xstart':
                target = xstart
            elif self.model_mean_type == 'eps':
                target = noise
            else:
                raise NotImplementedError(f"Error: Invalid model_mean_type. Expected 'xprev' or 'xstart' or 'eps' but got {self.model_mean_type}")
            output = self.denoiser(xt, t, )
            losses = torch.mean((output - target)**2, dim=tuple(range(1, len(xt.shape))))
        else:
            raise NotImplementedError(f"Error: Invalid loss_type. Expected 'kl' or 'mse' but got {self.loss_type}")

        return losses

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, _, _, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device)
        losses = self.training_losses(x, t)
        return losses.mean()

    def size_in_memory(self) -> None:
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 2**30
        print('Model size: {:.3f}GB'.format(size_all_mb))