import numpy as np
import torch
import ptwt
import pywt


class MinMaxScalar():
    def __init__(self):
        self.original_min = 0
        self.original_max = 0
    
    def __call__(self, x):
        self.original_min = x.min(axis=-1, keepdims=True)
        self.original_max = x.max(axis=-1, keepdims=True)
        
        x = x - self.original_min
        x = x / (self.original_max - self.original_min + 1e-6)
        
        return x

    
class ZScoreScalar():
    def __init__(self):
        self.original_mean = 0
        self.original_std = 0
    
    def __call__(self, x):
        self.original_mean = x.mean(axis=-1, keepdims=True)
        self.original_std = x.std(axis=-1, keepdims=True)
        
        x = x - self.original_mean
        x = x / (self.original_std + 1e-6)
        
        return x

    
def gaussian_p(n):
    def gaussian(x, mu=0.5, sig=0.5):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    p = np.array([gaussian(i) for i in np.linspace(0, 1, num=n)])
    p /= np.sum(p)
    
    return p

class RandomCrop():
    def __init__(self, length: int, validate: bool = False):
        self.length = length
        self.validate = validate

    def __call__(self, lead_data):
        max_startpoint = lead_data.shape[-1] - self.length
        start_point = np.random.randint(0, max_startpoint) if not self.validate else 0
        return lead_data[..., start_point:start_point + self.length]

    def __repr__(self):
        return f'{self.__class__.__name__}(length={self.length})'

class RandomLengthCrop():
    def __init__(self, min_length, max_length, validate: bool = False):
        self.min_length = min_length
        self.max_length = max_length
        self.validate = validate

    def __call__(self, lead_data):
        length = np.random.randint(self.min_length, self.max_length)
        max_startpoint = lead_data.shape[-1] - length
        start_point = np.random.randint(0, max_startpoint) if not self.validate else 0
        
        lead_data = lead_data[..., start_point:start_point + length]
        lead_data = np.pad(lead_data, ((0, 0), (0, self.max_length-length)), 'constant', constant_values=(0, 0))
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(max_length={self.max_length}, min_length={self.min_length})'

    
class DWTReconstruction():
    def __init__(self, wavelet_name='db5', level=4):
        self.wavelet_name = wavelet_name
        self.level = level
        assert self.wavelet_name in pywt.wavelist(), "Unknown wavelet name."
        
        self.gaussian_p = gaussian_p(self.level + 1)
    
    def __call__(self, x):  # x.shape=(batch, channels, time)
        # detect the shape of the input, expect shape: (channels, length)
        # reshape the input if needed
        ori_shape = x.shape
        reshaped = False
        if len(x.shape)==3:
            x = sig = x.flatten(0, 1)  # x.shape = (batch * channels, time)
            reshaped = True
        
        # performing DWT decomposition
        wavelet = pywt.Wavelet(self.wavelet_name)
        dwt_coeff = ptwt.wavedec(x, wavelet, mode='zero', level=self.level)
        
        # randomly select the coefficients to be masked
        for i in np.random.choice(self.level+1, 2, replace=False, p=self.gaussian_p):
            dwt_coeff[i] = dwt_coeff[i] * 0
        
        # reconstruct the signal from the masked coefficient set
        rec_ecg = ptwt.waverec(dwt_coeff, wavelet)
        
        # reshape the output if needed
        if reshaped:
            rec_ecg = rec_ecg.reshape((-1, ori_shape[-2], rec_ecg.shape[-1]))
        
        # return the result
        return rec_ecg
    
    def __repr__(self):
        return f'{self.__class__.__name__}(wavelet_name={self.wavelet_name}, level={self.level})'
    
    
class BaselineFiltering():
    def __init__(self, wavelet_name='db5', level=4):
        self.wavelet_name = wavelet_name
        self.level = level
        assert self.wavelet_name in pywt.wavelist(), "Unknown wavelet name."
    
    def __call__(self, x):  # x.shape=(batch, channels, time)
        # detect the shape of the input, expect shape: (channels, length)
        # reshape the input if needed
        ori_shape = x.shape
        reshaped = False
        if len(x.shape)==3:
            x = sig = x.flatten(0, 1)  # x.shape = (batch * channels, time)
            reshaped = True
        
        # performing DWT decomposition
        wavelet = pywt.Wavelet(self.wavelet_name)
        dwt_coeff = ptwt.wavedec(x, wavelet, mode='zero', level=self.level)
        
        # randomly select the coefficients to be masked
        for i in range(self.level):
            dwt_coeff[i+1] = dwt_coeff[i+1] * 0
        
        # reconstruct the signal from the masked coefficient set
        rec_ecg = ptwt.waverec(dwt_coeff, wavelet)
        
        # reshape the output if needed
        if reshaped:
            rec_ecg = rec_ecg.reshape((-1, ori_shape[-2], rec_ecg.shape[-1]))
        
        # return the result
        return rec_ecg
    
    def __repr__(self):
        return f'{self.__class__.__name__}(wavelet_name={self.wavelet_name}, level={self.level})'
    
    
class ChannelWiseDifference():
    def __init__(self):
        pass
    
    def __call__(self, x):
        # expect input shape: (C, L) or (N, C, L)
        c = x.shape[-2]
        x_ = x.clone()
        
        if len(x.shape)==2:  # (C, L)
            for i in range(c):
                x_[i] -= x[(i+1)%c]
                
        elif len(x.shape)==3:  # (N, C, L)
            for i in range(c):
                x_[:, i] -= x[:, (i+1)%c]
            
        return x_
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    
class AmplitudeScaling():
    def __init__(self, low=0.5, high=2):
        self.low = low
        self.high = high
    
    def __call__(self, x):
        scale = np.random.uniform(self.low, self.high)
        return x * scale
    
    def __repr__(self):
        return f'{self.__class__.__name__}(low={self.low}, high={self.high})'
        
        
class AmplitudeReversing():
    def __init__(self):
        pass
    
    def __call__(self, x):
        return x * -1
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
        
        
class GaussianNoise():
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    
    def __call__(self, x):
        noise = self.sigma * torch.randn(*x.shape)
        
        return x + noise.to(x.device)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma})'

    
class FlipX():
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.flip(x, dims=(-1,))
    
    def __repr__(self):
        return f'{self.__class__.__name__}'
    
    
class TimeOut():
    def __init__(self, max_mask_percentage=0.25):
        self.max_mask_percentage = max_mask_percentage
        assert 0 <= self.max_mask_percentage <= 1
    
    def __call__(self, x):
        # x.shape = (N, C, L) or (C, L)
        sig_len = x.shape[-1]
        mask_len = np.random.randint(0, int(sig_len * self.max_mask_percentage))
        
        max_start = sig_len - mask_len
        mask_start = np.random.randint(0, max_start)
        
        mask = torch.ones(*x.shape).to(x.device)
        mask[..., mask_start: mask_start+mask_len] = 0
        
        return x * mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}(max_mask_percentage={self.max_mask_percentage})'
    
    
class RandomLeadMask():
    def __init__(self, max_masked_leads=4):
        self.max_masked_leads = max_masked_leads
        assert 0 <= self.max_masked_leads <= 12
        
    def __call__(self, x):
        # x.shape = (N, C, L) or (C, L)
        num_masked_leads = np.random.randint(self.max_masked_leads+1)
        masked_leads = np.random.choice(12, num_masked_leads, replace=False)
        
        out = x.clone()
        out[..., masked_leads, :] = 0
        
        return out
    
    def __repr__(self):
        return f'{self.__class__.__name__}(max_masked_leads={self.max_masked_leads})'
    
    
class RandomSelectAugmentation():
    def __init__(self, n=2, augmentations=[# ChannelWiseDifference(), 
                                           AmplitudeScaling(), 
                                           # AmplitudeReversing(),
                                           GaussianNoise(),
                                           TimeOut(),
                                           RandomLeadMask()]):
        self.n = n
        self.augmentations = augmentations
        self.num_augs = len(augmentations)
        
        assert self.n <= self.num_augs
        
    def __call__(self, x):
        for i in np.random.choice(np.arange(self.num_augs), size=self.n, replace=False):
            x = self.augmentations[i](x)
        
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n}, num_augs={self.num_augs}, augmentations={self.augmentations})'