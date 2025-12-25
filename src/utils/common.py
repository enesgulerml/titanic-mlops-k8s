import yaml
import os


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'params.yaml')

    params = read_params(config_path)
    print(params)