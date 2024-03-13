from dataclasses import dataclass, field
import numpy as np
from scipy.fft import fftshift, fft2, ifftshift, fft
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

@dataclass(repr=False)
class Bispectrum2D:
    '''
    Make Bispectrum dataclass from 1D signal with frequency in Hertz.
    
    Parameters:
    -----------
        signal: The 1-Dimensional signal in (n,) numpy.ndarray
        freqsample:  The sample frequency in Hertz.  
        window: 'None', 'hanning', 'triangular'
    
    References
    ----------
    Matteo Bachetti, et al., stingray v1.0 code, DOI: https://zenodo.org/record/6394742, 2022.
    
    '''    
    signal: np.ndarray
    freqsample: float
    window_name: str
    dt: float = field(init=False)
    n: int = field(init=False)  #number of data points in signal
    maxlag: int = field(init=False)
    lagindex: np.ndarray = field(init=False)
    cum3_dim: int = field(init=False)
            
    def __post_init__(self):
        self.dt = 1 / self.freqsample        
        self.n = self.signal.shape[0]
        self.maxlag = int(self.n/2)
        self.lagindex = np.arange(-self.maxlag, self.maxlag + 1)
        self.cum3_dim = 2 * self.maxlag + 1
        self._calc_bispectrum()
        
    def _calc_bispectrum(self):
        self._cumulant3()
        self._window()
        self._bispectrum()
    
    def _cumulant3(self):
        '''Biased cumulant estimate.'''
        self.cum3 = np.zeros((self.cum3_dim, self.cum3_dim))  #include zeros matrix to reset calc 
        ind = np.arange((self.n - self.maxlag)-1, self.n) #consecutive idx from (n-maxlag-1) to n
        ind_t = np.arange(self.maxlag, self.n)
        zero_maxlag = np.zeros((1, self.maxlag))
        zero_maxlag_t = zero_maxlag.T
        sig = np.reshape(self.signal, (1,len(self.signal)))  #Reshape original self.sig
        sig = sig - np.mean(sig)                       #sig is 1xn row vector of counts.
        rev_sig = np.array([sig[0][::-1]])
        col = np.concatenate((sig.T[ind], zero_maxlag_t), axis=0)
        row = np.concatenate((rev_sig[0][ind_t], zero_maxlag[0]), axis=0)
        toep = toeplitz(col, row)
        rev_sig_repeat = np.repeat(rev_sig, [2 * self.maxlag + 1], axis=0) #n repeats
        #toep is n x (n-1).  It must be square to be a circulant.
        self.cum3 = (self.cum3 + np.matmul(np.multiply(toep, rev_sig_repeat), toep.T)) / self.n        
                
    def _window(self):        
        n = np.arange(self.cum3_dim)         #Total wind data points matches cum3_dim        
        self.window = np.zeros(self.cum3_dim)  
        if self.window_name == 'None':
            return
        if self.window_name == 'hanning':
            hanning = 0.5 * (1 - np.cos(2 * np.pi * n / (self.cum3_dim-1)))
            wind2D = np.tile(hanning,(self.cum3_dim,1))  #Make 2D wind by repeating rows N times.
            self.window[:self.maxlag + 1] = hanning[self.maxlag:]        
        if self.window_name == 'triangular':
            N_div_2 = int((np.floor((self.cum3_dim - 1) / 2)))
            triangular = 1 - np.abs((n - (N_div_2)) / self.cum3_dim)            
            wind2D = np.tile(triangular,(self.cum3_dim,1))  #Make 2D wind by repeating rows N times.
            self.window[:self.maxlag + 1] = triangular[self.maxlag:]    
        self.window[self.maxlag:] = 0
        # Put wind in toeplitz.  Each row of final wind is sliding hanning.
        row = np.concatenate(([self.window[0]], np.zeros(2 * self.maxlag)))
        toep_matrix = toeplitz(self.window, row)
        toep_matrix += np.tril(toep_matrix, -1).transpose()
        self.window = toep_matrix[..., ::-1] * wind2D * wind2D.T
    
    def _bispectrum(self):
        if self.window_name == 'None':
            self.bispec = fftshift(fft2(ifftshift(self.cum3)))
        else:
            self.bispec = fftshift(fft2(ifftshift(self.cum3*self.window)))         
        self.freqvals = 0.5 * self.freqsample * self.lagindex / self.maxlag        
        self.bispec_mag = np.abs(self.bispec)
        self.bispec_phase = np.angle(self.bispec)
        
    def plot_cum3(self):
        lags = self.lagindex * self.dt
        fig, ax1 = plt.subplots(1,1,figsize=(6,6)) #gist has (11,11)
        contplot1 = ax1.contourf(lags, lags, self.cum3, levels=100, cmap=plt.cm.Spectral_r)
        ax1.set_title('Third Order Cumulant'); ax1.set_xlabel('lag 1 values'); ax1.set_ylabel('lags 2 values');
    
    def plot_bispec_magnitude(self,new_file_path):    
        fig, ax1 = plt.subplots(1,1,figsize=(6,6))
        contplot1 = ax1.contourf(self.freqvals, self.freqvals, self.bispec_mag, levels=100, cmap=plt.cm.Spectral_r)
        plt.axis('off')  # Turn off axis
        plt.savefig(new_file_path, bbox_inches='tight', pad_inches=0) 
        # plt.close()

        return  ax1.contourf(self.freqvals, self.freqvals, self.bispec_mag, levels=100, cmap=plt.cm.Spectral_r)
