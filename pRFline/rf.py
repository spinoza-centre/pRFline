
import numpy as np

default_npix = 500
default_ndeg = 5
class PrfInfo(object):
    def __init__(self, x_pos, n_pix=default_npix, n_deg=default_ndeg):
        """            
        Parameters
        ----------
        x_pos : float, centre of Rf/s 
        n_pix : number of pixels
        n_deg : number of degrees (+/-)
        ------
        Returns
        -------
        None.
        """
        self.x_pos = x_pos
        self.n_pix = n_pix
        self.n_deg = n_deg
        self.x_grid = np.linspace(-n_deg, n_deg, n_pix)
        

    def return_pt_profile(self):
        '''
        Sometimes models are nonlinear (e.g., CSS, divisive norm), so it is hard to estimate 
        there response profile. This function tries to solve this by creating the response of the model to a series of 1d stim
        '''
        # pt_stim [time, pix] -> we want a "bar" of 1 pixel moving from L2R. So just an identity matrix
        self.pt_stim = np.reshape(np.eye(self.n_pix), (1, self.n_pix, self.n_pix))
        self.pt_resp = self.create_response(self.pt_stim)      
        self.pt_resp /= np.abs(self.pt_resp).max(-1)[...,np.newaxis]

    def return_bar_resp(self, n_steps, width):                
        self.n_steps = n_steps
        self.width = width

        # Make the stimulus
        bar_x = np.linspace(-self.n_deg, self.n_deg, self.n_steps)
        stim = np.zeros((self.n_steps, self.n_pix))
        for i,x in enumerate(bar_x):
            bar_start = x-width/2
            bar_end = x+width/2
            stim[i,:] = (self.x_grid>bar_start) & (self.x_grid<=bar_end)
        self.bar_x = bar_x
        self.stim = np.reshape(stim, (1, self.n_steps, self.n_pix)) # n_voxel, n_timept, n_pix
        self.bar_resp = self.create_response(self.stim)        
        self.bar_resp /= np.abs(self.bar_resp).max(-1)[...,np.newaxis]
        
    def _reshape_rf(self, rf): # so that: n_voxel, n_timept, n_pix
        r_rf = np.reshape(rf, (rf.shape[0], 1, rf.shape[1]))
        return r_rf

class CmPrfInfo(PrfInfo):
    def __init__(self, x_pos, a_sigma, a_val, n_pix=default_npix, n_deg=default_ndeg):
        super().__init__(x_pos=x_pos, n_pix=n_pix, n_deg=n_deg)
        self.a_sigma = a_sigma
        self.a_val = a_val
        self.name = "CM"
        self.aRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.a_sigma)       
        self.r_aRF = self._reshape_rf(self.aRF)
        self.pt_resp = self.return_pt_profile()  
    
    def create_response(self, stim):
        resp = self.a_val[...,np.newaxis] * np.sum(stim * self.r_aRF,-1)
        return resp

class CssPrfInfo(PrfInfo):
    def __init__(self, x_pos, a_sigma, a_val, n_val, n_pix=default_npix, n_deg=default_ndeg):
        super().__init__(x_pos=x_pos, n_pix=n_pix, n_deg=n_deg)
        self.a_sigma = a_sigma
        self.a_val = a_val
        self.n_val = n_val
        self.name = "CSS"
        self.aRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.a_sigma)     
        self.r_aRF = self._reshape_rf(self.aRF)        
        self.pt_resp = self.return_pt_profile()      
    
    def create_response(self, stim):
        resp = self.a_val[...,np.newaxis] * np.sum(stim * self.r_aRF,-1)**self.n_val[...,np.newaxis]
        return resp

class DogPrfInfo(PrfInfo):

    def __init__(self, x_pos, a_sigma, s_sigma, a_val, c_val, n_pix=default_npix, n_deg=default_ndeg):
        super().__init__(x_pos=x_pos, n_pix=n_pix, n_deg=n_deg)
        self.a_sigma = a_sigma
        self.s_sigma = s_sigma
        self.a_val = a_val
        self.c_val = c_val
        self.name = "Dog"
        self.aRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.a_sigma)                 
        self.r_aRF = self._reshape_rf(self.aRF)        
        self.sRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.s_sigma)     
        self.r_sRF = self._reshape_rf(self.sRF)        
        self.pt_resp = self.return_pt_profile()              
    
    def create_response(self, stim):
        # Activating input 
        resp = self.a_val[...,np.newaxis] * np.sum(stim * self.r_aRF,-1) - \
            self.c_val[...,np.newaxis] * np.sum(stim * self.r_sRF,-1)
        return resp


class DnPrfInfo(PrfInfo):

    def __init__(self, x_pos, a_sigma, s_sigma, a_val, b_val, c_val, d_val, n_pix=default_npix, n_deg=default_ndeg):
        super().__init__(x_pos=x_pos, n_pix=n_pix, n_deg=n_deg)
        self.a_sigma = a_sigma
        self.s_sigma = s_sigma
        self.a_val = a_val
        self.b_val = b_val
        self.c_val = c_val
        self.d_val = d_val
        self.name = "DN"
        self.aRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.a_sigma)                 
        self.r_aRF = self._reshape_rf(self.aRF)        
        self.sRF = make_gauss1d(x=self.x_grid, mu = self.x_pos, sigma=self.s_sigma)     
        self.r_sRF = self._reshape_rf(self.sRF)
        self.return_pt_profile()                
    
    def create_response(self, stim):
        # Activating input 
        a_inp = self.a_val[...,np.newaxis] * np.sum(stim * self.r_aRF,-1) + self.b_val[...,np.newaxis]
        # Normalising input
        n_inp = self.c_val[...,np.newaxis] * np.sum(stim * self.r_sRF,-1) + self.d_val[...,np.newaxis]
        # ~ model response
        resp = (a_inp/n_inp) - (self.b_val[...,np.newaxis]/self.d_val[...,np.newaxis])  
        return resp

        
def make_gauss1d(x, mu, sigma):
    """make_gauss1D
    takes 1-dimensional array x, containing the x coordinates at which to
    evaluate the gaussian function, with a given sigma, and returns a 1D array 
    Parameters
    ----------
    x : numpy.ndarray, 1D or flattened by masking
        , containing x coordinates
    mu : float: mean, coordinate of gauss 
    sigma : float, standard deviation of gauss
    Returns 
    -------
    numpy.ndarray, 1D gaussian values evaluated at (x)
    """

    gauss1d = np.exp(-((x-mu[...,np.newaxis])**2) /(2*(sigma[...,np.newaxis]**2)))

    return gauss1d