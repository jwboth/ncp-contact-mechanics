"""Utilities for dynamically composing contact mechanics models.

This module provides a registry-based approach to dynamically create model classes
with different contact mechanics formulations. Instead of long match/case statements,
formulations are registered in dictionaries and composed at runtime.

"""

from enum import StrEnum
from pathlib import Path
from typing import Type

import tomli

import ncp
import ncp.formulations


class NormalContactMechanicsFormulation(StrEnum):
    """Available normal contact mechanics formulations."""

    alart_curnier = "alart_curnier"
    constant_scaled_alart_curnier = "constant_scaled_alart_curnier"
    random_scaled_alart_curnier = "random_scaled_alart_curnier"
    decaying_scaled_alart_curnier = "decaying_scaled_alart_curnier"
    ncp_min = "ncp_min"
    ncp_min_mu = "ncp_min_mu"
    ncp_fb = "ncp_fb"
    ncp_fb_mu = "ncp_fb_mu"
    bipotential_orthogonal_return = "bipotential_orthogonal_return"
    constant_weighted_return = "constant_weighted_return"
    random_weighted_return = "random_weighted_return"
    soccp = "soccp"


class TangentialContactMechanicsFormulation(StrEnum):
    """Available tangential contact mechanics formulations."""

    hueber = "hueber"
    alart_curnier = "alart_curnier"
    constant_scaled_alart_curnier = "constant_scaled_alart_curnier"
    random_scaled_alart_curnier = "random_scaled_alart_curnier"
    decaying_scaled_alart_curnier = "decaying_scaled_alart_curnier"
    ncp_min = "ncp_min"
    ncp_min_mu = "ncp_min_mu"
    ncp_fb = "ncp_fb"
    bipotential_orthogonal_return = "bipotential_orthogonal_return"
    constant_weighted_return = "constant_weighted_return"
    random_weighted_return = "random_weighted_return"
    soccp = "soccp"


def _get_normal_contact_mixins() -> dict[
    NormalContactMechanicsFormulation, Type | None
]:
    """Registry mapping normal formulation names to mixin classes.

    Returns a dict where None means 'no mixin needed' (use base class directly).
    This is a function to avoid circular import issues at module load time.
    """
    return {
        NormalContactMechanicsFormulation.alart_curnier: None,
        NormalContactMechanicsFormulation.constant_scaled_alart_curnier: ncp.formulations.ConstantScaledAlartCurnier_NormalContact,
        NormalContactMechanicsFormulation.random_scaled_alart_curnier: ncp.formulations.RandomScaledAlartCurnier_NormalContact,
        NormalContactMechanicsFormulation.decaying_scaled_alart_curnier: ncp.formulations.DecayingScaledAlartCurnier_NormalContact,
        NormalContactMechanicsFormulation.ncp_min: ncp.formulations.NCP_MIN_NormalContact,
        NormalContactMechanicsFormulation.ncp_min_mu: ncp.formulations.NCP_MIN_MU_NormalContact,
        NormalContactMechanicsFormulation.ncp_fb: ncp.formulations.NCP_FB_NormalContact,
        NormalContactMechanicsFormulation.ncp_fb_mu: ncp.formulations.NCP_FB_MU_NormalContact,
        # Combined formulations - handled separately via COMBINED_CONTACT_MIXINS
        NormalContactMechanicsFormulation.bipotential_orthogonal_return: None,
        NormalContactMechanicsFormulation.constant_weighted_return: None,
        NormalContactMechanicsFormulation.random_weighted_return: None,
        NormalContactMechanicsFormulation.soccp: None,
    }


def _get_tangential_contact_mixins() -> dict[
    TangentialContactMechanicsFormulation, Type | None
]:
    """Registry mapping tangential formulation names to mixin classes.

    Returns a dict where None means 'no mixin needed' (use base class directly).
    This is a function to avoid circular import issues at module load time.
    """
    return {
        TangentialContactMechanicsFormulation.hueber: None,
        TangentialContactMechanicsFormulation.alart_curnier: ncp.formulations.RadialReturnTangentialContact,
        TangentialContactMechanicsFormulation.constant_scaled_alart_curnier: ncp.formulations.ConstantScaledRadialReturnTangentialContact,
        TangentialContactMechanicsFormulation.random_scaled_alart_curnier: ncp.formulations.RandomScaledRadialReturnTangentialContact,
        TangentialContactMechanicsFormulation.decaying_scaled_alart_curnier: ncp.formulations.DecayingScaledRadialReturnTangentialContact,
        TangentialContactMechanicsFormulation.ncp_min: ncp.formulations.NCP_MIN_TangentialContact,
        TangentialContactMechanicsFormulation.ncp_min_mu: ncp.formulations.NCP_MIN_MU_TangentialContact,
        TangentialContactMechanicsFormulation.ncp_fb: ncp.formulations.NCP_FB_TangentialContact,
        # Combined formulations - handled separately via COMBINED_CONTACT_MIXINS
        TangentialContactMechanicsFormulation.bipotential_orthogonal_return: None,
        TangentialContactMechanicsFormulation.constant_weighted_return: None,
        TangentialContactMechanicsFormulation.random_weighted_return: None,
        TangentialContactMechanicsFormulation.soccp: None,
    }


def _get_combined_contact_mixins() -> dict[tuple[str, str], Type]:
    """Registry for combined formulations that handle both normal and tangential.

    These formulations must be used for both normal and tangential contact together.
    This is a function to avoid circular import issues at module load time.
    """
    return {
        (
            "bipotential_orthogonal_return",
            "bipotential_orthogonal_return",
        ): ncp.formulations.BipotentialOrthogonalReturnContact,
        (
            "constant_weighted_return",
            "constant_weighted_return",
        ): ncp.formulations.ConstantWeightedReturnContact,
        (
            "random_weighted_return",
            "random_weighted_return",
        ): ncp.formulations.RandomWeightedReturnContact,
        ("soccp", "soccp"): ncp.formulations.SOCCPContactMechanics,
    }


# Set of formulation names that must be used for both normal and tangential
COMBINED_FORMULATIONS = {
    "bipotential_orthogonal_return",
    "constant_weighted_return",
    "random_weighted_return",
    "soccp",
}


def set_contact_mechanics_formulation(
    BaseModel: Type,
    normal_formulation: NormalContactMechanicsFormulation | str,
    tangential_formulation: TangentialContactMechanicsFormulation | str,
) -> Type:
    """Dynamically create a model class with the specified contact formulations.

    Parameters:
        BaseModel: The base model class to extend.
        normal_formulation: Normal contact formulation identifier.
        tangential_formulation: Tangential contact formulation identifier.

    Returns:
        A new class combining BaseModel with the appropriate contact mixins.

    Raises:
        ValueError: If an unknown formulation is specified or if combined
            formulations are not used consistently for both normal and tangential.

    """
    # Convert to enum if string
    if isinstance(normal_formulation, str):
        normal_formulation = NormalContactMechanicsFormulation(normal_formulation)
    if isinstance(tangential_formulation, str):
        tangential_formulation = TangentialContactMechanicsFormulation(
            tangential_formulation
        )

    # Load registries
    normal_mixins = _get_normal_contact_mixins()
    tangential_mixins = _get_tangential_contact_mixins()
    combined_mixins = _get_combined_contact_mixins()

    # Check for combined formulations first
    combined_key = (normal_formulation.value, tangential_formulation.value)
    if combined_key in combined_mixins:
        combined_mixin = combined_mixins[combined_key]
        return type(
            f"{BaseModel.__name__}WithContact",
            (combined_mixin, BaseModel),
            {},
        )

    # Validate that combined formulations are used together
    if normal_formulation.value in COMBINED_FORMULATIONS:
        raise ValueError(
            f"{normal_formulation.value} must be used for both normal and "
            f"tangential contact, but tangential is {tangential_formulation.value}."
        )
    if tangential_formulation.value in COMBINED_FORMULATIONS:
        raise ValueError(
            f"{tangential_formulation.value} must be used for both normal and "
            f"tangential contact, but normal is {normal_formulation.value}."
        )

    # Validate formulations exist in registry
    if normal_formulation not in normal_mixins:
        raise ValueError(f"Unknown normal formulation: {normal_formulation}")
    if tangential_formulation not in tangential_mixins:
        raise ValueError(f"Unknown tangential formulation: {tangential_formulation}")

    # Get mixins from registry
    normal_mixin = normal_mixins[normal_formulation]
    tangential_mixin = tangential_mixins[tangential_formulation]

    # Build inheritance chain: (tangential_mixin, normal_mixin, BaseModel)
    bases: list[Type] = []
    if tangential_mixin is not None:
        bases.append(tangential_mixin)
    if normal_mixin is not None:
        bases.append(normal_mixin)
    bases.append(BaseModel)

    # Create new class dynamically
    return type(
        f"{BaseModel.__name__}WithContact",
        tuple(bases),
        {},
    )


def add_contact_mechanics(Model: Type, input: Path | dict) -> Type:
    """Create a contact mechanics model from a config file or dictionary.

    Parameters:
        Model: The base model class.
        input: Path to TOML config file or config dictionary.

    Returns:
        A new model class with contact mechanics mixins.

    Raises:
        ValueError: If input type is not Path or dict.
        AssertionError: If input Path does not have .toml suffix.

    """
    if isinstance(input, Path):
        assert input.suffix == ".toml", f"Expected .toml file, got {input.suffix}"
        with open(input, "rb") as f:
            config = tomli.load(f)
    elif isinstance(input, dict):
        config = input
    else:
        raise ValueError(f"Expected Path or dict, got {type(input).__name__}")

    # Read formulations from config
    normal_formulation = config["contact"]["normal_formulation"]
    tangential_formulation = config["contact"]["tangential_formulation"]

    # Compose and return new model class
    return set_contact_mechanics_formulation(
        Model, normal_formulation, tangential_formulation
    )


class ContactConfig:
    """Configuration utilities for contact mechanics formulations."""

    @classmethod
    def default_config(cls) -> dict:
        """Default contact mechanics configuration."""
        return {
            "contact": {
                "normal_formulation": NormalContactMechanicsFormulation.alart_curnier.value,
                "tangential_formulation": TangentialContactMechanicsFormulation.alart_curnier.value,
            }
        }

    @classmethod
    def parse_model_params_from_config(cls, config: dict, model_params: dict) -> dict:
        """Parse contact mechanics parameters from config into model_params.

        Parameters:
            config: Configuration dictionary.
            model_params: Existing model parameters dictionary to update.

        Returns:
            Updated model_params dictionary with contact mechanics parameters.
        """
        model_params["contact"] = {}
        for key in ["normal_formulation", "tangential_formulation"]:
            if key not in config.get("contact", {}):
                raise KeyError(f"Missing contact.{key} in configuration.")
            model_params["contact"][key] = config["contact"][key]
        return model_params
