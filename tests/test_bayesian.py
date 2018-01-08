from __future__ import print_function
from orphics import maps,io,cosmology,lensing,stats,mpi
from enlib import enmap,bench
import numpy as np
import os,sys
from szar import counts
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("GridName", type=str,help='Name of directory to save cinvs to.')
parser.add_argument("GridMin", type=float,help='Min amplitude.')
parser.add_argument("GridMax", type=float,help='Max amplitude.')
parser.add_argument("GridNum", type=int,help='Number of amplitudes.')
parser.add_argument("-a", "--arc",     type=float,  default=10.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.5,help="Pix width (arcmin).")
parser.add_argument("-b", "--beam",     type=float,  default=1.0,help="Beam (arcmin).")
parser.add_argument("-s", "--simulate-cutout", action='store_true',help='Simulate unlensed cutouts instead of analytic covmat.')
parser.add_argument("-f", "--buffer-factor",     type=int,  default=2,help="Buffer factor for stamp.")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()

# Paths
PathConfig = io.load_path_config()
GridName = PathConfig.get("paths","output_data")+args.GridName

# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

# Geometry
shape, wcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=args.pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
bshape, bwcs = maps.rect_geometry(width_arcmin=args.arc*args.buffer_factor,px_res_arcmin=args.pix,pol=False)
bmodlmap = enmap.modlmap(bshape,bwcs)
bmodrmap = enmap.modrmap(bshape,bwcs)
# Unlensed signal
    
power2d = theory.uCl('TT',bmodlmap)
bfcov = maps.diagonal_cov(power2d)
sny,snx = shape
ny,nx = bshape
Ucov = maps.pixcov(bshape,bwcs,bfcov)
Ucov = Ucov.reshape(np.prod(bshape),np.prod(bshape))

# Noise model
kbeam = maps.gauss_beam(args.beam,bmodlmap)


# Lens template
lens_order = 5
posmap = enmap.posmap(bshape,bwcs)


# Lens grid
amin = args.GridMin
amax = args.GridMax
num_amps = args.GridNum
kamps = np.linspace(amin,amax,num_amps)


# MPI calculate set up
Nsims = num_amps
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]

# File I/O
io.mkdir(GridName,comm)
cov_name = lambda x: GridName+"/cov_"+str(x)+".npy"

if rank==0: print("Rank 0 starting ...")
for k,my_task in enumerate(my_tasks):
    kamp = kamps[my_task]


    kappa_template = lensing.nfw_kappa(kamp*1e15,bmodrmap,cc,overdensity=200.,critical=True,atClusterZ=True)
    phi,_ = lensing.kappa_to_phi(kappa_template,bmodlmap,return_fphi=True)
    grad_phi = enmap.grad(phi)
    pos = posmap + grad_phi
    alpha_pix = enmap.sky2pix(bshape,bwcs,pos, safe=False)


    def do_the_thing():
        return lensing.lens_cov(Ucov,alpha_pix,lens_order=lens_order,kbeam=kbeam,bshape=shape)

    if rank==0:
        with bench.show("rank 0 lensing cov"):
            Scov = do_the_thing()
    else:
        Scov = do_the_thing()
        
    np.save(cov_name(my_task),Scov)


if rank==0:
    io.save_cols(GridName+"/amps.txt",(kamps,))
    import json
    save_dict = {"arc":args.arc,"pix":args.pix,"beam":args.beam}
    with open(GridName+"/attribs.json",'w') as f:
        f.write(json.dumps(save_dict))
