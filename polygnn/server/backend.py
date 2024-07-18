from typing import List
import itertools as itt
import pandas as pd
import torch
import polygnn
import polygnn_trainer as pt
from pathlib import Path
from typing import Tuple


all_properties = {
    "mechanical": ["exp_TS__MPa", "exp_YM__GPa"],
    "thermal": [
        "exp_Tm__K",
        "exp_Tg__K",
        "exp_thermal_decomposition_temperature__K",
        "exp_thermal_conductivity__W_per_m_per_K",
    ],
    "electronic": [
        "DFT_bandgap_LF__eV",
        "DFT_bandgap_HF__eV",
        "DFT_EA__eV",
        "DFT_Eionization__eV",
    ],
    "optical_and_dielectric": [
        "exp_refractive_index",
        "DFT_refractive_index",
        "DFT_dielectric_total",
        "exp_dielectric_constant_1.78",
        "exp_dielectric_constant_2.0",
        "exp_dielectric_constant_3.0",
        "exp_dielectric_constant_4.0",
        "exp_dielectric_constant_5.0",
        "exp_dielectric_constant_6.0",
        "exp_dielectric_constant_7.0",
        "exp_dielectric_constant_9.0",
        "exp_dielectric_constant_15.0",
    ],
    "thermodynamic_and_physical": [
        "exp_Cp__J_per_gK",
        "DFT_Eatomization__eV_per_atom",
        "exp_limiting_oxygen_index__percentage",
        "exp_crystallinity_HF",
        "exp_crystallinity_LF",
        "exp_rho__g_per_cc",
        "exp_free_fractional_volume",
    ],
    "solubility_and_permeability": [
        "exp_solubility__MPa**0.5",
        "exp_perm_CH4__Barrer",
        "exp_perm_CO2__Barrer",
        "exp_perm_He__Barrer",
        "exp_perm_N2__Barrer",
        "exp_perm_O2__Barrer",
        "exp_perm_H2__Barrer",
    ],
}


currentdir = Path(__file__).resolve().parent
ROOT = currentdir / ".." /".."/ "trained_models" # path to "trained_models" directory in PolyGNN repository

# For convenience, let's define a function that makes predictions.
def _make_prediction_routine(data, dir_name):
    """
    Return the mean and std. dev. of a model prediction.

    Args:
        data (pd.DataFrame): The input data for the prediction.
        dir_name (str): The name of the directory containing the model that
            you desire to get predictions from. (e.g., "thermal", "electronic", etc.)
    """
    bond_config = polygnn.featurize.BondConfig(True, True, True)
    atom_config = polygnn.featurize.AtomConfig(
        True,
        True,
        True,
        True,
        True,
        True,
        combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
        aromatic=True,
    )

    root_dir = str(ROOT / dir_name)
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

    # Load scalers
    scaler_dict = pt.load2.load_scalers(root_dir)

    # Load selectors
    selectors = pt.load2.load_selectors(root_dir)

    # Load and evaluate ensemble.
    ensemble = pt.load.load_ensemble(
        root_dir,
        polygnn.models.polyGNN,
        device,
        {
            "node_size": atom_config.n_features,
            "edge_size": bond_config.n_features,
            "selector_dim": len(selectors),
        },
    )

    # Define a lambda function for smiles featurization.
    smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
        x,
        bond_config,
        atom_config,
        "monocycle",
    )

    # Perform inference
    y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=data,
        smiles_featurizer=smiles_featurizer,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )
    return y_mean_hat, y_std_hat


def properties_from_SMILES(
    smiles: List[str],
    models: List[str] = [
        "mechanical",
        "thermal",
        "electronic",
        "optical_and_dielectric",
        "thermodynamic_and_physical",
        "solubility_and_permeability",
    ],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute properties for a list of SMILES strings using Rishi Gurnani's PolyGNN models.

    Args:
        smiles (List[str]): SMILES strings for polymers.
        models (List[str], optional): list of models (classes of material properties) to apply. Options: "mechanical", "thermal", "electronic", "optical_and_dielectric", "thermodynamic_and_physical", "solubility_and_permeability". Defaults to all.

    Returns:
        pd.DataFrame: dataframe of computed properties.
        pd.DataFrame: dataframe of standard deviations of computed properties.
    """
    df = None
    df_stds = None
    for model_name in models:
        inputdf = pd.DataFrame(
            [
                {"smiles_string": s, "prop": p}
                for s, p in itt.product(smiles, all_properties[model_name])
            ]
        )

        # dfData[model_name] = {
        #     "input": inputdf
        # }
        inputdf["mean"], inputdf["std"] = _make_prediction_routine(inputdf, model_name)
        means = inputdf.pivot(index="smiles_string", columns="prop", values="mean")
        stds = inputdf.pivot(index="smiles_string", columns="prop", values="std")
        # df_ = (
        #     pd.merge(left=means, right=stds, on="smiles_string", suffixes=("", "_std"))
        #     .rename_axis(columns=None)
        #     .reset_index()
        # )
        if df is None:
            df = means
            df_stds = stds
        else:
            df = pd.merge(df, means, on="smiles_string")
            df_stds = pd.merge(df_stds, stds, on="smiles_string")
    return df, df_stds