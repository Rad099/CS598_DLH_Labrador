from lab_transformers.models.labrador import Labrador
import yaml

with open("configs/labrador.yaml", "r") as f:
    params = yaml.safe_load(f)

model = Labrador(params)

print(type(model))
