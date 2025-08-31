# TODO collect all available formulations by key
from enum import StrEnum
import ncp


class NormalContactMechanicsFormulation(StrEnum):
    alart_curnier = "alart_curnier"
    constant_scaled_alart_curnier = "CONSTANT_SCALED_ALART_CURNIER".lower()
    random_scaled_alart_curnier = "RANDOM_SCALED_ALART_CURNIER".lower()
    decaying_scaled_alart_curnier = "DECAYING_SCALED_ALART_CURNIER".lower()
    ncp_min = "NCP_MIN".lower()
    ncp_min_mu = "NCP_MIN_MU".lower()
    ncp_fb = "NCP_FB".lower()
    ncp_fb_mu = "NCP_FB_MU".lower()
    bipotential_orthogonal_return = "BIPOTENTIAL_ORTHOGONAL_RETURN".lower()
    constant_weighted_return = "CONSTANT_WEIGHTED_RETURN".lower()
    random_weighted_return = "RANDOM_WEIGHTED_RETURN".lower()
    soccp = "SOCCP".lower()


class TangentialContactMechanicsFormulation(StrEnum):
    hueber = "HUEBER".lower()
    alart_curnier = "ALART_CURNIER".lower()
    constant_scaled_alart_curnier = "CONSTANT_SCALED_ALART_CURNIER".lower()
    random_scaled_alart_curnier = "RANDOM_SCALED_ALART_CURNIER".lower()
    decaying_scaled_alart_curnier = "DECAYING_SCALED_ALART_CURNIER".lower()
    ncp_min = "NCP_MIN".lower()
    ncp_fb = "NCP_FB".lower()
    bipotential_orthogonal_return = "BIPOTENTIAL_ORTHOGONAL_RETURN".lower()
    constant_weighted_return = "CONSTANT_WEIGHTED_RETURN".lower()
    random_weighted_return = "RANDOM_WEIGHTED_RETURN".lower()
    soccp = "SOCCP".lower()


def make_contact_mechanics_model(
    BaseModel,
    normal_formulation: NormalContactMechanicsFormulation,
    tangential_formulation: TangentialContactMechanicsFormulation,
):
    match normal_formulation:
        ### Alart Curnier type formulations
        case NormalContactMechanicsFormulation.alart_curnier:
            # Special case of constant scaled alart curnier, where the scaling exponent is 0.

            class ModelWithNormalContact(BaseModel): ...

        case NormalContactMechanicsFormulation.constant_scaled_alart_curnier:

            class ModelWithNormalContact(
                ncp.formulations.ConstantScaledAlartCurnier_NormalContact, BaseModel
            ): ...

        case NormalContactMechanicsFormulation.random_scaled_alart_curnier:

            class ModelWithNormalContact(
                ncp.formulations.RandomScaledAlartCurnier_NormalContact, BaseModel
            ): ...

        case NormalContactMechanicsFormulation.decaying_scaled_alart_curnier:

            class ModelWithNormalContact(
                ncp.formulations.DecayingScaledAlartCurnier_NormalContact, BaseModel
            ): ...

        ### NCP formulations

        case NormalContactMechanicsFormulation.ncp_min:

            class ModelWithNormalContact(
                ncp.formulations.NCP_MIN_NormalContact, BaseModel
            ): ...

        case NormalContactMechanicsFormulation.ncp_min_mu:

            class ModelWithNormalContact(
                ncp.formulations.NCP_MIN_MU_NormalContact, BaseModel
            ): ...

        case NormalContactMechanicsFormulation.ncp_fb:

            class ModelWithNormalContact(
                ncp.formulations.NCP_FB_NormalContact, BaseModel
            ): ...

        case NormalContactMechanicsFormulation.ncp_fb_mu:

            class ModelWithNormalContact(
                ncp.formulations.NCP_FB_MU_NormalContact, BaseModel
            ): ...

        ### Bipotential formulations

        case NormalContactMechanicsFormulation.bipotential_orthogonal_return:
            assert (
                tangential_formulation
                == TangentialContactMechanicsFormulation.bipotential_orthogonal_return
            ), (
                "Bipotential orthogonal return can only be used for both normal and tangential contact."
            )

            # Model will be setup below.
            ...

        ### Combined projections

        case NormalContactMechanicsFormulation.constant_weighted_return:
            assert (
                tangential_formulation
                == TangentialContactMechanicsFormulation.constant_weighted_return
            ), (
                "Constant weighted return can only be used for both normal and tangential contact."
            )

            # Model will be setup below.
            ...

        case "random_weighted_return":
            assert tangential_formulation == "random_weighted_return", (
                "Random weighted return can only be used for both normal and tangential contact."
            )

            # Model will be setup below.
            ...

        case _:
            raise ValueError(f"Unknown normal formulation: {normal_formulation}")

    match tangential_formulation:
        ### Alart Curnier type formulations

        case TangentialContactMechanicsFormulation.hueber:
            # Special case of constant scaled alart curnier, where the scaling exponent is 1.

            class ModelWithTangentialContact(ModelWithNormalContact): ...

        case TangentialContactMechanicsFormulation.alart_curnier:
            # Special case of constant scaled alart curnier, where the scaling exponent is 0.

            class ModelWithTangentialContact(
                ncp.formulations.RadialReturnTangentialContact, ModelWithNormalContact
            ): ...

        case TangentialContactMechanicsFormulation.constant_scaled_alart_curnier:

            class ModelWithTangentialContact(
                ncp.formulations.ConstantScaledRadialReturnTangentialContact,
                ModelWithNormalContact,
            ): ...

        case TangentialContactMechanicsFormulation.random_scaled_alart_curnier:

            class ModelWithTangentialContact(
                ncp.formulations.RandomScaledRadialReturnTangentialContact,
                ModelWithNormalContact,
            ): ...

        case TangentialContactMechanicsFormulation.decaying_scaled_alart_curnier:

            class ModelWithTangentialContact(
                ncp.formulations.DecayingScaledRadialReturnTangentialContact,
                ModelWithNormalContact,
            ): ...

        ### NCP formulations

        case TangentialContactMechanicsFormulation.ncp_min:

            class ModelWithTangentialContact(
                ncp.formulations.NCP_MIN_TangentialContact, ModelWithNormalContact
            ): ...

        case TangentialContactMechanicsFormulation.ncp_fb:

            class ModelWithTangentialContact(
                ncp.formulations.NCP_FB_TangentialContact, ModelWithNormalContact
            ): ...

        ### Bipotential formulations

        case TangentialContactMechanicsFormulation.bipotential_orthogonal_return:
            assert normal_formulation == "bipotential_orthogonal_return", (
                "Bipotential orthogonal return can only be used for both normal and tangential contact."
            )

            class ModelWithTangentialContact(
                ncp.formulations.BipotentialOrthogonalReturnContact, BaseModel
            ): ...

        ### Combined projections

        case TangentialContactMechanicsFormulation.constant_weighted_return:
            assert normal_formulation == "constant_weighted_return", (
                "Constant weighted return can only be used for both normal and tangential contact."
            )

            class ModelWithTangentialContact(
                ncp.formulations.ConstantWeightedReturnContact, BaseModel
            ): ...

        case TangentialContactMechanicsFormulation.random_weighted_return:
            assert normal_formulation == "random_weighted_return", (
                "Random weighted return can only be used for both normal and tangential contact."
            )

            class ModelWithTangentialContact(
                ncp.formulations.RandomWeightedReturnContact, BaseModel
            ): ...

        case _:
            raise ValueError(
                f"Unknown tangential formulation: {tangential_formulation}"
            )

    Model = ModelWithTangentialContact

    return Model
