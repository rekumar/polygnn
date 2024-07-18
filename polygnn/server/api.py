from fastapi import FastAPI
from .datamodels import InferenceInput
from .backend import properties_from_SMILES

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

# Step 7: Run the FastAPI app using Uvicorn
# Run the following command in your terminal:
# uvicorn api:app --reload