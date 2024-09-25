import sys

from poli.core.registry import DEFAULT_OBSERVER_NAME

from lolbo.poli_objective import PoliObjective
from scripts.molecule_optimization import MoleculeOptimization

sys.path.append("../")
import fire
from lolbo.molecule_objective import MoleculeObjective
from lolbo.utils.mol_utils.load_data import compute_train_zs, load_molecule_train_data
from poli import create


class PoliOptimization(MoleculeOptimization):
    def __init__(self, **kwargs):
        task_id = kwargs['task_id']
        self.problem = create(name=task_id, observer_name=DEFAULT_OBSERVER_NAME)
        kwargs['task_id'] = 'logp'
        super().__init__(**kwargs)

    def initialize_objective(self):
        # initialize molecule objective
        self.objective = PoliObjective(
            problem=self.problem,
            task_id=self.task_id,
            path_to_vae_statedict=self.path_to_vae_statedict,
            max_string_length=self.max_string_length,
            smiles_to_selfies=self.init_smiles_to_selfies
        )
        # if train zs have not been pre-computed for particular vae, compute them 
        #   by passing initialization selfies through vae 
        if self.init_train_z is None:
            self.init_train_z = compute_train_zs(
                self.objective,
                self.init_train_x,
            )
        return self

    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        assert self.num_initialization_points <= 20_000
        # smiles, selfies, zs, ys = load_molecule_train_data(
        #     task_id=self.task_id,
        #     num_initialization_points=self.num_initialization_points,
        #     path_to_vae_statedict=self.path_to_vae_statedict
        # )
        smiles = self.problem.x0.tolist()
        smiles = ["".join(s) for s in smiles]
        ys = self.problem.black_box(self.problem.x0)
        zs = None  # will be computed
        self.init_train_x, self.init_train_z, self.init_train_y = smiles, zs, ys
        if self.verbose:
            print("Loaded initial training data")
            print("train y shape:", self.init_train_y.shape)
            print(f"train x list length: {len(self.init_train_x)}\n")

        # create initial smiles to selfies dict
        self.init_smiles_to_selfies = {}
        return self
        for ix, smile in enumerate(self.init_train_x):
            self.init_smiles_to_selfies[smile] = selfies[ix]

        return self



if __name__ == "__main__":
    fire.Fire(PoliOptimization)
