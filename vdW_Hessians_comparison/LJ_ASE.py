from ase.calculators.lj import LennardJones
from ase.io import read, write
from ase.optimize import BFGS

# R0 in Bohr
R0_vdW = {'H': 3.1000, 'He': 2.6500, 'Li': 4.1600, 'Be': 4.1700, 'B': 3.8900, 'C': 3.5900,
            'N': 3.3400, 'O': 3.1900, 'F': 3.0400, 'Ne': 2.9100, 'Na': 3.7300, 'Mg': 4.2700,
            'Al': 4.3300, 'Si': 4.2000, 'P': 4.0100, 'S': 3.8600, 'Cl': 3.7100, 'Ar': 3.5500,
            'K': 3.7100, 'Ca': 4.6500, 'Sc': 4.5900, 'Ti': 4.5100, 'V': 4.4400, 'Cr': 3.9900,
            'Mn': 3.9700, 'Fe': 4.2300, 'Co': 4.1800, 'Ni': 3.8200, 'Cu': 3.7600, 'Zn': 4.0200,
            'Ga': 4.1900, 'Ge': 4.2000, 'As': 4.1100, 'Se': 4.0400, 'Br': 3.9300, 'Kr': 3.8200,
            'Rb': 3.7200, 'Sr': 4.5400, 'Y': 4.8151, 'Zr': 4.53, 'Nb': 4.2365, 'Mo': 4.099,
            'Tc': 4.076, 'Ru': 3.9953, 'Rh': 3.95, 'Pd': 3.6600, 'Ag': 3.8200, 'Cd': 3.99,
            'In': 4.2319, 'Sn': 4.3030, 'Sb': 4.2760, 'Te': 4.22, 'I': 4.1700, 'Xe': 4.0800,
            'Cs': 3.78, 'Ba': 4.77, 'La': 3.14, 'Ce': 3.26, 'Pr': 3.28, 'Nd': 3.3,
            'Pm': 3.27, 'Sm': 3.32, 'Eu': 3.40, 'Gd': 3.62, 'Tb': 3.42, 'Dy': 3.26,
            'Ho': 3.24, 'Er': 3.30, 'Tm': 3.26, 'Yb': 3.22, 'Lu': 3.20, 'Hf': 4.21,
            'Ta': 4.15, 'W': 4.08, 'Re': 4.02, 'Os': 3.84, 'Ir': 4.00, 'Pt': 3.92,
            'Au': 3.86, 'Hg': 3.98, 'Tl': 3.91, 'Pb': 4.31, 'Bi': 4.32, 'Po': 4.097,
            'At': 4.07, 'Rn': 4.23, 'Fr': 3.90, 'Ra': 4.98, 'Ac': 2.75, 'Th': 2.85,
            'Pa': 2.71, 'U': 3.00, 'Np': 3.28, 'Pu': 3.45, 'Am': 3.51, 'Cm': 3.47,
            'Bk': 3.56, 'Cf': 3.55, 'Es': 3.76, 'Fm': 3.89, 'Md': 3.93, 'No': 3.78}

#C6 in Hartree*Bohr^6
C6_vdW = {'H': 6.5000, 'He': 1.4600, 'Li': 1387.0000, 'Be': 214.0000, 'B': 99.5000, 'C': 46.6000,
        'N': 24.2000, 'O': 15.6000, 'F': 9.5200, 'Ne': 6.3800, 'Na': 1556.0000, 'Mg': 627.0000,
        'Al': 528.0000, 'Si': 305.0000, 'P': 185.0000, 'S': 134.0000, 'Cl': 94.6000, 'Ar': 64.3000,
        'K': 3897.0000, 'Ca': 2221.0000, 'Scv': 1383.0000, 'Ti': 1044.0000, 'V': 832.0000, 'Cr': 602.0000,
        'Mn': 552.0000, 'Fe': 482.0000, 'Co': 408.0000, 'Ni': 373.0000, 'Cu': 253.0000, 'Zn': 284.0000,
        'Ga': 498.0000, 'Ge': 354.0000, 'As': 246.0000, 'Se': 210.0000, 'Br': 162.0000, 'Kr': 129.6000,
        'Rb': 4691.0000, 'Sr': 3170.0000, 'Y': 1968.580, 'Zr': 1677.91, 'Nb': 1263.61, 'Mo': 1028.73,
        'Tc': 1390.87,
        'Ru': 609.754, 'Rh': 469.0, 'Pd': 157.5000, 'Ag': 339.0000, 'Cd': 452.0, 'In': 707.0460,
        'Sn': 587.4170,
        'Sb': 459.322, 'Te': 396.0, 'I': 385.0000, 'Xe': 285.9000, 'Cs': 6582.08, 'Ba': 5727.0, 'La': 3884.5,
        'Ce': 3708.33, 'Pr': 3911.84, 'Nd': 3908.75, 'Pm': 3847.68, 'Sm': 3708.69, 'Eu': 3511.71,
        'Gd': 2781.53, 'Tb': 3124.41, 'Dy': 2984.29, 'Ho': 2839.95, 'Er': 2724.12, 'Tm': 2576.78,
        'Yb': 2387.53, 'Lu': 2371.80, 'Hf': 1274.8, 'Ta': 1019.92, 'W': 847.93, 'Re': 710.2, 'Os': 596.67,
        'Ir': 359.1, 'Pt': 347.1, 'Au': 298.0, 'Hg': 392.0, 'Tl': 717.44, 'Pb': 697.0, 'Bi': 571.0,
        'Po': 530.92, 'At': 457.53, 'Rn': 390.63, 'Fr': 4224.44, 'Ra': 4851.32, 'Ac': 3604.41, 'Th': 4047.54,
        'Pa': 2367.42, 'U': 1877.10, 'Np': 2507.88, 'Pu': 2117.27, 'Am': 2110.98, 'Cm': 2403.22,
        'Bk': 1985.82,
        'Cf': 1891.92, 'Es': 1851.1, 'Fm': 1787.07, 'Md': 1701.0, 'No': 1578.18}



# Setting up the calculator
elem = "Ar"
ABOHR = 0.52917721 # in AA
HARTREE = 27.211383 # in eV
sigma = R0_vdW[elem] * (2**(-1/6.)) * ABOHR
epsilon = C6_vdW[elem]/((R0_vdW[elem]*ABOHR)**6) / 2. * HARTREE*(ABOHR**6)
calc = LennardJones(sigma=sigma, epsilon=epsilon)
# Reading atoms
atoms = read("0003.in", format = "aims")
atoms.set_calculator(calc)
# Perform relaxation
opt = BFGS(atoms, trajectory="LJ_Ar2.traj", logfile="log.log")
opt.run(fmax=0.001, steps=1000)
print("relaxed")
# Calculating of the Hessian with finite differences for the relaxed geometry
from ase.vibrations import Vibrations
import numpy as np
calc = LennardJones(sigma=sigma, epsilon=epsilon)
vib = Vibrations(atoms)
vib.clean()
vib.delta=0.005
vib.run()
vib.summary()
np.savetxt("finite_Hes_ase.hes", vib.H)

# Calculating of the Hessian analytically taken from GenSec
from gensec.precon import vdwHessian
vdW_hes = vdwHessian(atoms)
np.savetxt("vdW_Hes.hes", vdW_hes)

# Differences in Hessians
np.savetxt("diff.hes", vib.H-vdW_hes)

write("relaxed_0003.in", atoms, format="aims")
