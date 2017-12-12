import numpy as np
from enlib import enmap, coordinates


def random_catalog(shape,wcs,N,edge_avoid_deg=0.):

    box = enmap.box(shape,wcs)
    dec0 = min(box[0,0],box[1,0]) + edge_avoid_deg*np.pi/180.
    dec1 = max(box[0,0],box[1,0]) - edge_avoid_deg*np.pi/180.
    ra0 = min(box[0,1],box[1,1]) + edge_avoid_deg*np.pi/180.
    ra1 = max(box[0,1],box[1,1]) - edge_avoid_deg*np.pi/180.

    ras = np.random.uniform(ra0,ra1,N) * 180./np.pi
    decs = np.random.uniform(dec0,dec1,N) * 180./np.pi

    return ras,decs

class HealpixCatMapper(object):

    def __init__(self,nside,ras_deg,decs_deg):
        import healpy as hp
        print( "Calculating pixels...")
        self.pixs = hp.ang2pix(nside,ras_deg,decs_deg,lonlat=True)
        print( "Done with pixels...")
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.counts = self.get_map()

    def get_map(self,weights=None):
        print("Calculating histogram...")
        return np.histogram(self.pixs,bins=self.npix,weights=weights,range=[0,self.npix])[0]

    def _counts(self):
        cts = self.counts.copy()
        cts[self.mask<0.9] = np.nan
        self.ngals = np.nansum(cts)
        self.nmean = np.nanmean(cts)
        area_sqdeg = enmap.area(self.shape,self.wcs)*(180./np.pi)**2.
        self.frac = self.mask.sum()*1./self.mask.size
        self.area_sqdeg = self.frac*area_sqdeg
        self.ngal_per_arcminsq  = self.ngals/(self.area_sqdeg*60.*60.)

    def get_delta(self):
        return (self.counts/self.nmean-1.)
    


class CatMapper(object):

    def __init__(self,shape,wcs,ras_deg,decs_deg):
        coords = np.vstack((decs_deg,ras_deg))*np.pi/180.
        self.shape = shape
        self.wcs = wcs
        print( "Calculating pixels...")
        self.pixs = enmap.sky2pix(shape,wcs,coords,corner=True) # should corner=True?!
        print( "Done with pixels...")
        self.counts = self.get_map()

    def get_map(self,weights=None):
        Ny,Nx = self.shape
        print("Calculating histogram...")
        return enmap.ndmap(np.histogram2d(self.pixs[0,:],self.pixs[1,:],bins=self.shape,weights=weights,range=[[0,Ny],[0,Nx]])[0],self.wcs)

    def _counts(self):
        cts = self.counts.copy()
        cts[self.mask<0.9] = np.nan
        self.ngals = np.nansum(cts)
        self.nmean = np.nanmean(cts)
        area_sqdeg = enmap.area(self.shape,self.wcs)*(180./np.pi)**2.
        self.frac = self.mask.sum()*1./self.mask.size
        self.area_sqdeg = self.frac*area_sqdeg
        self.ngal_per_arcminsq  = self.ngals/(self.area_sqdeg*60.*60.)

    def get_delta(self):
        return (self.counts/self.nmean-1.)
    


class BOSSMapper(CatMapper):

    def __init__(self,shape,wcs,boss_files,random_files=None,rand_sigma_arcmin=2.,rand_threshold=1e-3,zmin=None,zmax=None):
        from astropy.io import fits

        ras = []
        decs = []
        for boss_file in boss_files:
            f = fits.open(boss_file)
            cat = f[1] #.copy()
            ras += cat.data['RA'].tolist()
            decs += cat.data['DEC'].tolist()
            f.close()
            
        CatMapper.__init__(self,shape,wcs,ras,decs)
        if random_files is not None:
            self.rand_map = 0.
            #ras = []
            #decs = []
            for random_file in random_files:
                print ("Opening fits...")
                f = fits.open(random_file)
                print ("Done opening fits...")
                cat = f[1] #.copy()
                ras = cat.data['RA'] #.tolist()
                decs = cat.data['DEC'] #.tolist()
                rcat = CatMapper(shape,wcs,ras,decs)
                self.rand_map += rcat.counts
                del rcat
                del ras
                del decs
                del cat
                f.close()
            self.update_mask(rand_sigma_arcmin,rand_threshold)

    def update_mask(self,rand_sigma_arcmin=2.,rand_threshold=1e-3):
        if rand_sigma_arcmin>1.e-3:
            print( "Smoothing...")
            smap = enmap.smooth_gauss(self.rand_map,rand_sigma_arcmin*np.pi/180./60.)
            print( "Done smoothing...")
        else:
            smap = self.rand_map
            
        self.mask = np.zeros(self.shape)
        self.mask[smap>rand_threshold] = 1
        self._counts()
            
    
class HSCMapper(CatMapper):

    def __init__(self,shape,wcs,cat_fits,pz_fits=None,zmin=None,zmax=None,mask_threshold=4.):
        from astropy.io import fits
        f = fits.open(cat_fits)
        self.cat = f[1].copy()
        f.close()
        ras = self.cat.data['ira']
        decs = self.cat.data['idec']
        self.wts = self.cat.data['ishape_hsm_regauss_derived_weight']
        if pz_fits is not None:
            raise NotImplementedError

        CatMapper.__init__(self,shape,wcs,ras,decs)
        self.hsc_wts = self.get_map(weights=self.wts)
        self.mean_wt = np.nan_to_num(self.hsc_wts/self.counts)
        self.update_mask(mask_threshold)

    def update_mask(self,mask_threshold):
        mask = np.zeros(self.shape)
        mask[self.mean_wt>mask_threshold] = 1
        self.mask = mask
        self._counts()

        
    def get_shear(self,do_m=True,do_c=True):
        rms = self.cat.data['ishape_hsm_regauss_derived_rms_e']
        m = self.cat.data['ishape_hsm_regauss_derived_bias_m']
        e1 = self.cat.data['ishape_hsm_regauss_e1']
        e2 = self.cat.data['ishape_hsm_regauss_e2']
        c1 = self.cat.data['ishape_hsm_regauss_derived_bias_c1']
        c2 = self.cat.data['ishape_hsm_regauss_derived_bias_c2']

        hsc_wts = self.hsc_wts
        wts = self.wts
        hsc_resp = 1.-np.nan_to_num(self.get_map(weights=(wts*(rms**2.))) / hsc_wts)
        hsc_m = np.nan_to_num(self.get_map(weights=(wts*(m))) / hsc_wts) if do_m else hsc_wts*0.

        hsc_e1 = self.get_map(weights=e1*wts)
        hsc_e2 = self.get_map(weights=e2*wts)

        hsc_c1 = np.nan_to_num(self.get_map(weights=c1*wts)/hsc_wts) if do_c else hsc_wts*0.
        hsc_c2 = np.nan_to_num(self.get_map(weights=c2*wts)/hsc_wts) if do_c else hsc_wts*0.

        g1map = np.nan_to_num(hsc_e1/2./hsc_resp/(1.+hsc_m)/hsc_wts) - np.nan_to_num(hsc_c1/(1.+hsc_m))
        g2map = np.nan_to_num(hsc_e2/2./hsc_resp/(1.+hsc_m)/hsc_wts) - np.nan_to_num(hsc_c2/(1.+hsc_m))

        return g1map,g2map

def split_samples(in_samples,split_points):
    """ Calculate statistics on splits of a sample of data.
    If in_samples is a list containing measurements of (say) richness
    and split_points are the richness values corresponding to bin 
    edges of the splits, this function will return the "S/N", 
    average richness and number of objects in each bin defined 
    by the split_points bin edges.

    Here, S/N is average richness times square root of number
    objects in that bin. If the real S/N is linear in richness
    and sqrt(N), this returned quantity should be proportional
    to it.


    e.g. 
    >>> in_samples = [5,1,6,7,2,9,10]
    >>> split_points = [1,5,10]
    >>> sns, means, Ns = split_points(in_samples,split_points)
    >>> print (Ns)
    >>> [3,4]
    

    """
    assert np.all(np.diff(split_points))>0., "Split points should be monotonically increasing."
    A = np.asarray(in_samples)
    sns = []
    means = []
    Ns = []
        
    for a,b in zip(split_points[:-1],split_points[1:]):
        loc = np.where(np.logical_and(A>a,A<=b))
        split_mean = A[loc].mean()
        means.append(split_mean)
        split_N = len(A[loc])
        Ns.append(split_N)
        sns.append(split_mean*np.sqrt(split_N))
        
        
    return np.asarray(sns),np.asarray(means),np.asarray(Ns)


def optimize_splits(in_samples,in_splits):
    """ Given measurements in_samples and bin edges in_splits,
    this function solves for a new set of bin edges out_splits
    (keeping the leftmost and rightmost edge fixed) such
    that the "S/N" in each bin is as close to each other as
    possible. (Concretely, it minimizes the variance of the
    S/N in each bin).

    See split_samples for a definition of S/N.

    """

    def cost(*kwargs):
        in_array = np.asarray(kwargs).flatten()
        if np.any(np.diff(in_array))<0.: return np.inf
        out_splits = np.append(np.append(in_splits[0],in_array),in_splits[-1]).flatten()
        sns,means,Ns = split_samples(in_samples,out_splits)
        return np.var(sns)


    from scipy.optimize import fmin
    in_splits = np.asarray(in_splits)
    res = fmin(cost,in_splits[1:-1])
    out_splits = np.append(np.append(in_splits[0],res),in_splits[-1]).flatten()
    return out_splits
