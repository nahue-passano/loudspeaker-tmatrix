from pydantic import BaseModel
import yaml


class FrequencyConfig(BaseModel):
    min: int
    max: int
    n_bins: int


class AcousticalConstantsConfig(BaseModel):
    sound_speed: float
    air_density: float
    atmospheric_pressure: int
    reference_pressure: float
    measurement_distance: float
    directivity_factor: int


class Config(BaseModel):
    default_loudspeaker_cfg: str
    frequency: FrequencyConfig
    acoustical_constants: AcousticalConstantsConfig


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)


class ElectricalConfig(BaseModel):
    input_voltage: float
    coil_resistance: float
    coil_inductance: float


class MechanicalConfig(BaseModel):
    mass: float
    compliance: float
    resistance: float


class AcousticalConfig(BaseModel):
    effective_diameter: float


class LoudspeakerConfig(BaseModel):
    electrical: ElectricalConfig
    electromechanical_factor: float
    mechanical: MechanicalConfig
    acoustical: AcousticalConfig


def load_loudspeaker_config(yaml_path: str) -> LoudspeakerConfig:
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return LoudspeakerConfig(**data)
