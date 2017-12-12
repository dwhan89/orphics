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

# Unlensed signal
power2d = theory.uCl('TT',modlmap)
fcov = maps.diagonal_cov(power2d)
Ucov = maps.pixcov(shape,wcs,fcov).reshape(np.prod(shape),np.prod(shape))

# Noise model
kbeam = maps.gauss_beam(args.beam,modlmap)


# Lens template
kappa_template = lensing.nfw_kappa(1e15,modrmap,cc)
phi,_ = lensing.kappa_to_phi(kappa_template,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
lens_order = 5
posmap = enmap.posmap(shape,wcs)


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
    pos = posmap + kamp*grad_phi
    alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)


    def do_the_thing():
        return lensing.lens_cov(Ucov,alpha_pix,lens_order=lens_order,kbeam=kbeam)

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
