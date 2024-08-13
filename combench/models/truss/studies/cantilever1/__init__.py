from combench.models.truss.studies.cantilever1.problems import train_problems
from combench.models.truss.studies.cantilever1.problems import val_problems
from combench.models.truss.studies.cantilever1.problems import oob_validation_problems as oob_problems
from combench.models import truss

# ------------------------------------------------
# Set Norms
# ------------------------------------------------

for p in train_problems:
    truss.set_norms(p)
for p in val_problems:
    truss.set_norms(p)
for p in oob_problems:
    truss.set_norms(p)

# --- Common Norms
# max_train_norm = max([p['norms'][0] for p in train_problems])
# max_train_volfrac = max([p['volfrac_norms'][1] for p in train_problems])
# for p in train_problems:
#     p['norms'][0] = max_train_norm
#     p['volfrac_norms'][0] = 0
#     p['volfrac_norms'][1] = max_train_volfrac
# for p in val_problems:
#     p['norms'][0] = max_train_norm
#     p['volfrac_norms'][0] = 0
#     p['volfrac_norms'][1] = max_train_volfrac
# for p in val_problems_out:
#     p['norms'][0] = max_train_norm
#     p['volfrac_norms'][0] = 0
#     p['volfrac_norms'][1] = max_train_volfrac






































