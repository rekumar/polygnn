from .backend import all_properties
from typing import List, Optional
from pydantic import BaseModel, validator


class InferenceInput(BaseModel):
    """Example Pydantic model for the input data.
    """
    smiles: List[str]
    property_types: Optional[List[str]] = list(all_properties.keys())
 
    
    @validator("smiles", each_item=True)
    def validate_polymer_smiles(cls, v):
        if not isinstance(v, str):
            raise ValueError("Each SMILES entry must be a string.")
        if not(v.count("*") == 2):
            raise ValueError("Each polymer SMILES string must contain exactly two '*' characters. The asterisks represent the connection point between individual monomers shown in the SMILES string.")
        return v
    
    @validator("property_types", each_item=True)
    def validate_property_types(cls, v):
        if not isinstance(v, str):
            raise ValueError("Each property type must be a string.")
        if v not in all_properties:
            raise ValueError(f"Invalid property type: {v}. Must be one of {all_properties}.")
        return v