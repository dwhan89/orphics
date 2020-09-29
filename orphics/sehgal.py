import healpy as hp
import numpy as np
import os,sys
from past.utils import old_div
from pixell import enmap, utils

default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

def jysr2thermo(nu, tcmb=default_tcmb):
    nu = np.asarray(nu)*1e9
    mu = H_CGS*(nu)/(K_CGS*tcmb)
    conv_fact = 2*(K_CGS*tcmb)**3/(H_CGS**2*C_light**2)*mu**4/(4*(np.sinh(mu/2.))**2)
    conv_fact *= 1e23
    return 1/conv_fact*tcmb*1e6

class SehgalSky(object):
    def __init__(self,path=None,healpix=True): 
        if path is None: path = path = os.environ['SEHGAL_SKY']
        if path[-1]!='/': path = path + '/' 
        self.path        = path
        self.healpix     = healpix
        self.sim_postfix = '' if healpix else '_car_1_arcmin'
        self.frequencies = [str(s).zfill(3) for s in [27,30,39,44,70,93,100,143,145,217,225,280,353]]
        self.areas = [4000,8000,16000]
        if healpix:
            self.rfunc = hp.read_map
            self.nside = 4096
        else:
            self.rfunc = enmap.read_map
            self.shape,self.wcs = enmap.read_map_geometry(self.get_lensed_cmb(True))

    def get_total_cmb(self,freq,filename_only=False):
        freq     = str(freq).zfill(3)
        filename = self.path + "%s_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp%s.fits" %(freq, self.sim_postfix) 
        return filename if filename_only else self.rfunc(filename)

    def get_lensed_cmb(self,filename_only=False):
        filename = self.path + "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map%s.fits" %(self.sim_postfix) 
        return filename if filename_only else self.rfunc(filename)

    def get_ksz(self,filename_only=False):
        filename = self.path + "148_ksz_healpix_nopell_Nside4096_DeltaT_uK%s.fits" %self.sim_postfix
        return filename if filename_only else self.rfunc(filename)

    def get_kappa(self,filename_only=False):
        filename = self.path + "healpix_4096_KappaeffLSStoCMBfullsky%s.fits" %self.sim_postfix
        return filename if filename_only else self.rfunc(filename)

    def get_compton_y(self,filename_only=False):
        filename = self.path + "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75%s.fits" %self.sim_postfix
        return filename if filename_only else self.rfunc(filename)

    def get_cib(self,freq,filename_only=False):
        freq     = str(freq).zfill(3)
        filename = self.path + "%s_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75%s.fits" %(freq, self.sim_postfix)
        return filename if filename_only else self.rfunc(filename)

    def get_radio(self,freq,filename_only=False):
        freq     = str(freq).zfill(3)
        filename = self.path + "%s_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp%s.fits" %(freq,self.sim_postfix)
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_dust(self,freq,filename_only=False):
        freq     = str(freq).zfill(3)
        filename = self.path + "%s_dust_healpix_nopell_Nside4096_DeltaT_uK_lininterp%s.fits" %(freq,self.sim_postfix)
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_lf(self,freq,filename_only=False):
        freq     = str(freq).zfill(3)
        filename = self.path + "commander_synch_freefree_AME_%s_Equ_uK_smooth17arcmin%s.fits" %(freq, self.sim_postfix)
        return filename if filename_only else self.rfunc(filename)

    def get_mask(self,area,filename_only=False):
        area = str(int(area)).zfill(5)
        filename = self.path + "masks/mask_%s_Equ_bool_Nside4096%s.fits" % (area,self.sim_postfix)
        return filename if filename_only else self.rfunc(filename)

    def cmb_ps(self):
        ells,ps = np.loadtxt(self.path+"CMB_PS_healpix_Nside4096_DeltaT_uK_SimLensCMB.txt",unpack=True)
        return ells,ps

    def ksz_ps(self):
        ells,ps = np.loadtxt(self.path+"kSZ_PS_Sehgal_healpix_Nside4096_DeltaT_uK.txt",unpack=True)
        return ells,ps

    def compton_y_ps(self):
        ells,ps = np.loadtxt(self.path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
        return ells,ps

    def tsz_ps(self,freq,comptony=None,tcmb=default_tcmb):
        ells,ps = np.loadtxt(self.path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
        return ells, ps * (fnu(freq)* tcmb * 1e6)**2.

    def get_tsz(self,freq,comptony=None,tcmb=default_tcmb):
        if comptony is None: comptony = self.get_compton_y()
        return fnu(freq) * comptony * tcmb * 1e6




class SehgalSky2010(object):
    # to handle orignal Sehgal et al sims
    def __init__(self,path=None, data_type="healpix", unit="thermo"): 
        if path is None: path = os.environ['SEHGAL_SKY']
        assert(data_type in ['healpix', 'enmap', 'alm'])
        self.data_type = data_type
        self.path        = path
        self.unit        = unit
        assert(self.unit in ["thermo", "jysr"])
        self.frequencies = [str(s).zfill(3) for s in [30,90,148,219,277,350]]
        if self.data_type == 'healpix' :
            self.rfunc = hp.read_map
            self.nside = 8192
        elif self.data_type == 'enmap':
            self.rfunc = enmap.read_map
            self.shape,self.wcs = enmap.fullsky_geometry(res=0.5*utils.arcmin)
        elif self.data_type == 'alm':
            self.rfunc = lambda x : hp.read_alm(x, hdu=1)

    def __get_unit_conv(self, freq):
        ## mistake here healpix in jysr while others in thermo
        ret = None
        if self.data_type == "healpix":
            ret = 1. if self.unit == "jysr" else jysr2thermo(freq)
        else:
            ret = 1. if self.unit == "thermo" else 1./jysr2thermo(freq)
        return ret

    def __format_fname(self, fname_temp, freq=None):
        if freq is not None:
            freq = str(freq).zfill(3)
            fname = os.path.join(self.path, fname_temp.format(freq, self.data_type))
        else:
            fname = os.path.join(self.path, fname_temp.format(self.data_type))
        return fname

    def get_total_cmb(self,freq,filename_only=False):
        filename = self.__format_fname('{}_skymap_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename) * self.__get_unit_conv(freq)

    def get_lensed_cmb(self,freq, filename_only=False):
        filename = self.__format_fname('{}_lensedcmb_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename) * self.__get_unit_conv(freq)

    def get_ksz(self, freq, filename_only=False): 
        filename = self.__format_fname('{}_ksz_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename) * self.__get_unit_conv(freq)

    def get_kappa(self,filename_only=False): 
        filename = self.__format_fname('kappa_{}.fits', freq=None)
        return filename if filename_only else self.rfunc(filename)

    def get_compton_y(self, tcmb=default_tcmb, scale=0.75):
        return self.get_tsz(148, scale)/(fnu(148)*tcmb * 1e6) * self.__get_unit_conv(freq)

    def get_cib(self,freq,filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_ir_pts_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale * self.__get_unit_conv(freq)

    def get_radio(self,freq,filename_only=False):
        filename = self.__format_fname('{}_rad_pts_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename) * self.__get_unit_conv(freq)

    def get_galactic_dust(self,freq,filename_only=False): 
        filename = self.__format_fname('{}_dust_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename) * self.__get_unit_conv(freq)

    def get_tsz(self,freq,filename_only=False, scale=0.75):
        filename = self.__format_fname('{}_tsz_{}.fits', freq=freq)
        return filename if filename_only else self.rfunc(filename)*scale * self.__get_unit_conv(freq)
