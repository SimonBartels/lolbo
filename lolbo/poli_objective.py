import numpy as np
import poli
import torch 
import selfies as sf

from lolbo.molecule_objective import MoleculeObjective
from lolbo.utils.mol_utils.mol_utils import smiles_to_desired_scores
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from lolbo.utils.mol_utils.selfies_vae.data import collate_fn
from lolbo.latent_space_objective import LatentSpaceObjective
from lolbo.utils.mol_utils.mol_utils import GUACAMOL_TASK_NAMES
import pkg_resources
# make sure molecule software versions are correct: 
assert pkg_resources.get_distribution("selfies").version == '2.0.0'
assert pkg_resources.get_distribution("rdkit-pypi").version == '2022.3.1'
assert pkg_resources.get_distribution("molsets").version == '0.3.1'


class PoliObjective(MoleculeObjective):
    def __init__(self, problem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem = problem

    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assumes x is a single smiles string
        score = self.problem.black_box(x)

        return score
