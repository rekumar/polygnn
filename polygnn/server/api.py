from pathlib import Path
from fastapi import FastAPI
from .datamodels import InferenceInput
from .backend import properties_from_SMILES, all_properties
import json

thisdir = Path(__file__).parent

with open(thisdir / "properties.json", "r") as f:
    property_details = json.load(f)


app = FastAPI(
    title="PolyGNN API",
    description="""
    API for predicting polymer properties using pretrained PolyGNN models. 
    
    Properties include mechanical, thermal, electronic, optical and dielectric, thermodynamic and physical, and solubility and permeability.
    """,
)

@app.post(
    "/predict",
    summary = "Predict properties for a batch of polymers from SMILES string(s).",
    description = "Predict the properties of polymers based on their SMILES strings. Input data containing a list of SMILES strings and a list of property types to predict for. Each SMILES string must contain exactly two '*' characters indicating the connection point between individual monomers. Valid property types are ['mechanical', 'thermal', 'electronic', 'optical_and_dielectric', 'thermodynamic_and_physical', 'solubility_and_permeability'].",
    response_description = "Dictionary containing the computed properties. Properties are returned in a `pandas.DataFrame.to_json` format. Two dataframes are returned: one for the mean prediction values for each property and one for the standard deviations.",
    )
async def predict(input_data: InferenceInput):
    """Predict the properties of polymers based on their SMILES strings.

    Args:
        input_data (InferenceInput): Pydantic model for the input data.

    Returns:
        Dict: Dictionary containing the computed properties. Two dataframes are returned: one for the mean prediction values for each property and one for the standard deviations.
    """
    print(input_data.smiles)
    df, df_std = properties_from_SMILES(input_data.smiles)
    return {
        "means": df.to_json(),
        "stds": df_std.to_json()
    }


@app.get("/availableProperties", summary="Get a list of available property types.", description="Get a list of available property types that can be predicted using the PolyGNN models. The keys of the returned dictionary are the individual PolyGNN models trained on that class of properties. One or more of these property types can be used as input to the /predict endpoint; for each type, the list of corresponding properties will be predicted.")
async def available_properties():
    """Get a list of available property types.

    Returns:
        List: List of available property types.
    """
    detailed = {
        property_class: {
            prop: property_details[prop] 
            for prop in props}
        for property_class, props
        in all_properties.items()
        }
    
    return detailed