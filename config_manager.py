import yaml
import json
import logging
import os
from typing import Dict, Any
from threading import Lock

# Define constants and configuration
CONFIG_FILE = 'config.yaml'
CONFIG_BACKUP_FILE = 'config_backup.yaml'
PROGRAMS = ['Early Years', 'School-Based', 'School Years']

# Define exception classes
class ConfigError(Exception):
    """Base class for configuration-related exceptions."""
    pass

class ConfigLoadError(ConfigError):
    """Raised when loading configuration fails."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigUpdateError(ConfigError):
    """Raised when updating configuration fails."""
    pass

# Define data structures/models
class Config:
    """Represents a configuration for a clinical program."""
    def __init__(self, program: str, settings: Dict[str, Any]):
        self.program = program
        self.settings = settings

# Define validation functions
def validate_settings(settings: Dict[str, Any]) -> bool:
    """Validates the given settings dictionary."""
    required_keys = ['program_name', 'program_description']
    for key in required_keys:
        if key not in settings:
            return False
    return True

# Define utility methods
def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file into a dictionary."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Failed to load YAML file: {e}")
        raise ConfigLoadError("Failed to load YAML file")

def save_yaml_file(file_path: str, data: Dict[str, Any]) -> None:
    """Saves a dictionary to a YAML file."""
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
    except Exception as e:
        logging.error(f"Failed to save YAML file: {e}")
        raise ConfigUpdateError("Failed to save YAML file")

# Define the ConfigManager class
class ConfigManager:
    """Manages configuration for different clinical programs."""
    def __init__(self):
        self.configs = {}
        self.lock = Lock()

    def load_program_config(self, program: str) -> Config:
        """Loads the configuration for the given program."""
        with self.lock:
            if program not in self.configs:
                try:
                    config_data = load_yaml_file(CONFIG_FILE)
                    program_config = config_data.get(program, {})
                    if not validate_settings(program_config):
                        raise ConfigValidationError("Invalid configuration settings")
                    self.configs[program] = Config(program, program_config)
                except Exception as e:
                    logging.error(f"Failed to load program config: {e}")
                    raise ConfigLoadError("Failed to load program config")
            return self.configs[program]

    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validates the given settings dictionary."""
        return validate_settings(settings)

    def update_configurations(self, program: str, settings: Dict[str, Any]) -> None:
        """Updates the configuration for the given program."""
        with self.lock:
            try:
                if not self.validate_settings(settings):
                    raise ConfigValidationError("Invalid configuration settings")
                config_data = load_yaml_file(CONFIG_FILE)
                config_data[program] = settings
                save_yaml_file(CONFIG_FILE, config_data)
                self.configs[program] = Config(program, settings)
            except Exception as e:
                logging.error(f"Failed to update program config: {e}")
                raise ConfigUpdateError("Failed to update program config")

    def manage_environment_vars(self) -> None:
        """Manages environment variables for the configuration."""
        # Implement environment variable management logic here
        pass

    def backup_configurations(self) -> None:
        """Backs up the current configurations."""
        with self.lock:
            try:
                config_data = load_yaml_file(CONFIG_FILE)
                save_yaml_file(CONFIG_BACKUP_FILE, config_data)
            except Exception as e:
                logging.error(f"Failed to backup configurations: {e}")
                raise ConfigError("Failed to backup configurations")

# Define the main function
def main():
    config_manager = ConfigManager()
    program = 'Early Years'
    config = config_manager.load_program_config(program)
    print(f"Loaded configuration for {program}: {config.settings}")

    # Update configuration
    updated_settings = {'program_name': 'Updated Program', 'program_description': 'Updated description'}
    config_manager.update_configurations(program, updated_settings)
    print(f"Updated configuration for {program}: {config_manager.load_program_config(program).settings}")

    # Backup configurations
    config_manager.backup_configurations()
    print("Configurations backed up successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()