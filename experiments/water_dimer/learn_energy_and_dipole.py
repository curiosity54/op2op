from hamlet.utils.target_utils import get_blocks
from hamlet.utils.twocenter_utils import H_power_from_blocks
from hamlet.models.property_model import convert_twoctmap_to_struct
from hamlet.data.dataset import QuantumData 
from hamlet.features.acdc import single_center_features
from hamlet.models.property_model import convert_onectmap_to_struct
from ase.io import read
import os
import hickle
import torch 
import matplotlib.pyplot as plt
import metatensor.torch as mts
from sklearn.linear_model import RidgeCV
from hamlet.metrics import L2_loss
import numpy as np

BASIS  = 'ccpvtz'
TARGET_BASIS = 'def2-tzvp'
DEVICE = 'cpu'
model_name = "vext_dipole"
BASE_DIR = './'
model_path = './trained_models'

START = 0
STOP = 1000
jump = 1
data_slices = slice(START, STOP, jump)

frames = read(os.path.join(BASE_DIR, 'data/water_dimers/water_dimers.xyz'),data_slices) 
vext = hickle.load(os.path.join(BASE_DIR, f'data/water_dimers/{BASIS}/vext.hickle'))[data_slices]


qmdata = QuantumData(frames = frames,
                     data = {'vext':vext
                            },
                     fix_matrix_orbital_order=True,
                     basis = BASIS,
                     device = DEVICE
                    )

target_energies = hickle.load(os.path.join(BASE_DIR, f'data/water_dimers/{TARGET_BASIS}/energy.hickle'))[data_slices]
# these are [x,y,z]
target_dipoles = hickle.load(os.path.join(BASE_DIR, f'data/water_dimers/{TARGET_BASIS}/dipole.hickle'))[data_slices]
# make them [y,z,x]
target_dipoles = target_dipoles[:, [1,2,0]]

max_radial  = 6
max_angular = 4
atomic_gaussian_width = 0.3

hypers = { "cutoff": {
        "radius": 4,
        "smoothing": {
            "type": "ShiftedCosine",
            "width": 0.1
        }
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": max_angular,
        "radial": {
            "type": "Gto",
            "max_radial": max_radial
        }
    }
}


acdc = single_center_features(frames, hypers = hypers, order_nu = 2, lcut = 4)
acdc_str = convert_onectmap_to_struct(acdc, atoms_to_sum = torch.tensor([0,3]))
acdc_invariant_str = acdc_str.block({'spherical_harmonics_l':0, 'inversion_sigma':1})
acdc_L1_str = acdc_str.block({'spherical_harmonics_l':1, 'inversion_sigma':1})

# computed with all pairs = True
cblocks_vext_ = get_blocks(qmdata, matrix = qmdata.data['vext'], orbitals = qmdata.orbitals, device=DEVICE, all_pairs = True, orbitals_to_properties = True, skip_symmetry = True)
cblocks_v2 = H_power_from_blocks(qmdata, matrices =qmdata.data['vext'],  orbitals = qmdata.orbitals, power = 2, orbitals_to_properties = True, skip_symmetry = True, all_pairs = True)
cblocks_v3 = H_power_from_blocks(qmdata, matrices = qmdata.data['vext'], orbitals = qmdata.orbitals, power = 3, orbitals_to_properties = True, skip_symmetry = True, all_pairs = True)
cblocks_v4 = H_power_from_blocks(qmdata, matrices = qmdata.data['vext'], orbitals = qmdata.orbitals, power = 4, orbitals_to_properties = True, skip_symmetry = True, all_pairs = True)

cblocks_v123 = mts.join([cblocks_vext_, 
                          mts.multiply(cblocks_v2, 0.7), 
                          mts.multiply(cblocks_v2, 1.0), 
                          mts.multiply(cblocks_v3, 0.001),
                         mts.multiply(cblocks_v4, 0.0001),
                         ], 
                         axis = 'properties')

vext_struct = convert_twoctmap_to_struct(cblocks_v123, atoms_to_sum = torch.tensor([0,3]))
vext_invariant_str = vext_struct.block({'L':0, 'parity':1})
vext_L1_str = vext_struct.block({'L':1, 'parity':1})
# split data into training and validation sets by distance
def build_distances(
    distance_min=3.5,
    distance_max=12.0,
    distance_count=20,
    npair = 50
):
    base = np.linspace(distance_min, distance_max, distance_count)
    distances = []
    for ipair in range(npair):
        distances.extend(base)
    return distances

MAX_TRAIN_DIST = 8
distances = build_distances( )
train_idx = [i for i, dist in enumerate(distances) if dist < MAX_TRAIN_DIST]
train_idx += [i for i, dist in enumerate(distances) if dist == 12.0]

val_idx = np.setdiff1d(range(len(distances)), train_idx)
print(f'Selected {len(train_idx)} frames for training, {len(val_idx)} for val')
n_d = 20

# --- ACDC energy ---
alphas = np.logspace(-7, 3, 40)
x = acdc_invariant_str.values[train_idx].squeeze(1).numpy()
y = target_energies.numpy() [train_idx]
trainmean = y.mean()

y_mean = y - trainmean
ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(x,y_mean) 

pred = ridge.predict(x)
train_loss = L2_loss(torch.tensor(pred)+trainmean, torch.tensor(y))

x_val = acdc_invariant_str.values[val_idx].squeeze(1).numpy()
y_val = target_energies[val_idx].numpy()
pred_val = ridge.predict(x_val)
val_loss = L2_loss(torch.tensor(pred_val)+trainmean, torch.tensor(y_val))

print(f"Train loss : {torch.sqrt(train_loss/len(y))}")
print(f"Val loss : {torch.sqrt(val_loss/len(y_val))}")
print(f"ref std: {np.std(y)}")
x = acdc_invariant_str.values.squeeze(1).numpy()
pred_all = ridge.predict(x)
predE_acdc = pred_all.reshape(-1, n_d)
target = target_energies.reshape(-1, n_d)

# --- ACDC dipole ---
alphas = np.logspace(-4, 1, 40)
x = acdc_L1_str.values[train_idx].numpy()
nstruct, ncomp, nprop = x.shape 
x = x.reshape(nstruct* ncomp, -1)
y = target_dipoles.numpy()[train_idx]
ridge = RidgeCV(alphas=alphas, fit_intercept=False).fit(x,y.reshape(-1,1)) 

pred = ridge.predict(x).reshape(nstruct, ncomp)
train_loss = L2_loss(torch.tensor(pred), torch.tensor(y))

x_val = acdc_L1_str.values[val_idx].numpy()
nstruct_val = x_val.shape[0]
x_val = x_val.reshape(nstruct_val* ncomp, -1)
y_val = target_dipoles.numpy()[val_idx]

pred_val = ridge.predict(x_val).reshape(nstruct_val, ncomp)
val_loss = L2_loss(torch.tensor(pred_val), torch.tensor(y_val))

print(f"Train loss : {torch.sqrt(train_loss/len(y))}")
print(f"Val loss : {torch.sqrt(val_loss/len(y_val))}")
print(f"ref std: {np.std(y)}")

#---- Vext energy ---
alphas = np.logspace(-10, 4, 40)
# alphas = np.logspace(-5, 1, 40) # mp 
x = vext_invariant_str.values[train_idx].squeeze(1).numpy()
y = target_energies.numpy() [train_idx]
trainmean = y.mean()

y_mean = y - trainmean
ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(x,y_mean) 

pred = ridge.predict(x)
train_loss = L2_loss(torch.tensor(pred)+trainmean, torch.tensor(y))

x_val = vext_invariant_str.values[val_idx].squeeze(1).numpy()
y_val = target_energies.numpy()[val_idx]
pred_val = ridge.predict(x_val)
val_loss = L2_loss(torch.tensor(pred_val)+trainmean, torch.tensor(y_val))

print(f"Train loss : {torch.sqrt(train_loss/len(y))}")
print(f"Val loss : {torch.sqrt(val_loss/len(y_val))}")
print(f"ref std: {np.std(y)}")
x = vext_invariant_str.values.squeeze(1).numpy()
pred_all = ridge.predict(x)
predE = pred_all.reshape(-1, n_d)
#---- Vext dipole ---

alphas = np.logspace(-6, 1, 40)
# alphas = np.logspace(-4 -1, 40) # mp

x = vext_L1_str.values[train_idx].numpy()
nstruct, ncomp, nprop = x.shape 
x = x.reshape(nstruct* ncomp, -1)

y = target_dipoles.numpy()[train_idx]
ridge = RidgeCV(alphas=alphas, fit_intercept=False).fit(x,y.reshape(-1,1)) 

pred = ridge.predict(x).reshape(nstruct, ncomp)
train_loss = L2_loss(torch.tensor(pred), torch.tensor(y))

x_val = vext_L1_str.values[val_idx].numpy()
nstruct_val = x_val.shape[0]
x_val = x_val.reshape(nstruct_val* ncomp, -1)
y_val = target_dipoles.numpy()[val_idx]

pred_val = ridge.predict(x_val).reshape(nstruct_val, ncomp)
val_loss = L2_loss(torch.tensor(pred_val), torch.tensor(y_val))

print(f"Train loss : {torch.sqrt(train_loss/len(y))}")
print(f"Val loss : {torch.sqrt(val_loss/len(y_val))}")
print(f"ref std: {np.std(y)}")


# --- Plot results ---
from ase.units import Hartree
ha2ev = Hartree 
fig, ax = plt.subplots(1,1, figsize = (6,4))
ax = plt.gca()
ax.ticklabel_format(axis='y', style='plain', useOffset=False)
ax.plot(distances[:20], predE[19, :] * ha2ev, 'b.', label = r'$\mathbf{V}^{\kappa_\text{max}}$')
ax.plot(distances[:20], predE[19, :]* ha2ev, 'b-')
ax.plot(distances[:20], predE_acdc[19, :]* ha2ev, 'r.', label = r'$\rho_i^{\otimes 2}$')
ax.plot(distances[:20], predE_acdc[19, :]* ha2ev, 'r-')
ax.plot(distances[:20], (target[19,:]-trainmean)* ha2ev, 'k--', alpha = 0.9, label = 'reference')
ax.axvspan(xmin=x.min(), xmax=8, color='gray', alpha=0.2)
ax.set_xlabel(r"$d \,  (\AA)$", fontsize = 12)
ax.set_ylabel(r"$ E - \overline{E}\ \,\, (eV) $", fontsize = 12)
ax.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.savefig(
    "water_dimers.pdf",
    bbox_inches="tight",
    dpi=300
)
