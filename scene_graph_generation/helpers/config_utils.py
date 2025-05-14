import os
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigObject:
    """
    A class that wraps a dictionary and allows attribute access to values.
    This enables using config.key instead of config["key"] syntax.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self._config_dict = config_dict
        # Add all items as attributes
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access for compatibility.
        """
        return self._config_dict[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Allow 'in' operator.
        """
        return key in self._config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Provide a get method like dictionaries.
        """
        return self._config_dict.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert back to a dictionary.
        """
        return self._config_dict
    
    def __repr__(self) -> str:
        """
        String representation.
        """
        return f"ConfigObject({self._config_dict})"


class ConfigManager:
    # Static variable to store the current temporary config path
    _temp_config_path = None
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager with a path to the config file.
        
        Args:
            config_path (str, optional): Path to the config file. If None, will use
                the default config.yaml in the helpers directory.
        """
        if config_path is None:
            config_path = os.path.join(Path(__file__).parent, "config.yaml")
        self.config_path = config_path
        self.config_dict = self.load_config()
        self.config = ConfigObject(self.config_dict)
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str, optional): Path to the configuration file.
                If None, will use the path provided during initialization.
                
        Returns:
            dict: Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file is not found
            yaml.YAMLError: If config file cannot be parsed
        """
        path_to_use = config_path or self.config_path
        
        try:
            with open(path_to_use, 'r') as file:
                full_config = yaml.safe_load(file)
                # Extract the 'Config' key to maintain compatibility
                if 'Config' in full_config:
                    return full_config['Config']
                return full_config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {path_to_use}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
    
    def get_config(self) -> ConfigObject:
        """Return the current config object with attribute access."""
        return self.config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Return the current config as a dictionary."""
        return self.config_dict
    
    def update_config(self, updates: Dict[str, Any], save: bool = True) -> None:
        """
        Update config in memory and optionally save to file.
        
        Args:
            updates (dict): Dictionary with updates to apply to the config
            save (bool): Whether to save the updated config to file
        """
        def deep_update(d: Dict, u: Dict) -> Dict:
            """Recursively update nested dictionary."""
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        # Update in-memory config dictionary
        self.config_dict = deep_update(self.config_dict.copy(), updates)
        
        # Recreate the ConfigObject with the updated dictionary
        self.config = ConfigObject(self.config_dict)
        
        # Save to file if requested
        if save:
            self.save_config()
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save current config to file.
        
        Args:
            config_path (str, optional): Path to save the config.
                If None, will use the path provided during initialization.
                
        Raises:
            IOError: If the config cannot be saved
        """
        path_to_use = config_path or self.config_path
        
        try:
            # Wrap in 'Config' key for compatibility with original format
            full_config = {'Config': self.config_dict}
            
            with open(path_to_use, 'w') as file:
                yaml.safe_dump(full_config, file, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Error saving config to {path_to_use}: {e}")
    
    def write_to_temp_config(self) -> str:
        """
        Write the current config to a temporary file.
        
        Returns:
            str: Path to the temporary configuration file
            
        Raises:
            IOError: If the config cannot be saved to a temporary file
        """
        try:
            # Create a temporary file with .yaml extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
            temp_path = temp_file.name
            
            # Save the config to the temporary file
            self.save_config(temp_path)
            
            # Store the temp path in the class variable
            ConfigManager._temp_config_path = temp_path
            
            return temp_path
        except Exception as e:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(f"Error creating temporary config file: {e}")
    
    def write_to_fixed_temp_config(self, name="temp_config") -> str:
        """
        Write the current config to a fixed temporary file location.
        This allows other parts of the application to access the same temp config.
        
        Args:
            name (str): Base name for the temporary file (will be appended with .yaml)
            
        Returns:
            str: Path to the fixed temporary configuration file
        """
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{name}.yaml")
        
        try:
            # Save the config to the fixed temporary file
            self.save_config(temp_path)
            
            # Store the temp path in the class variable
            ConfigManager._temp_config_path = temp_path
            
            return temp_path
        except Exception as e:
            raise IOError(f"Error creating fixed temporary config file: {e}")
    
    @classmethod
    def get_last_temp_config_path(cls) -> Optional[str]:
        """
        Get the path to the most recently created temporary config file.
        
        Returns:
            str or None: Path to the last temporary config file, or None if none exists
        """
        return cls._temp_config_path


# For backwards compatibility with the functional approach
def load_config(config_path=None) -> ConfigObject:
    """
    Load configuration from YAML file and return as a ConfigObject with attribute access.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        
    Returns:
        ConfigObject: Configuration object with attribute access
    """
    config_manager = ConfigManager(config_path)
    return config_manager.get_config()


def load_config_dict(config_path=None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and return as a dictionary.
    
    Args:
        config_path (str, optional): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary
    """
    config_manager = ConfigManager(config_path)
    return config_manager.get_config_dict()


def write_to_temp_config(config: Union[Dict[str, Any], ConfigObject]) -> str:
    """
    Write a configuration to a temporary file.
    
    Args:
        config: Configuration dictionary or ConfigObject to write
        
    Returns:
        str: Path to the temporary configuration file
    """
    config_manager = ConfigManager()
    
    # Convert ConfigObject to dict if needed
    if isinstance(config, ConfigObject):
        config = config.to_dict()
        
    # Replace the loaded config with the provided one
    config_manager.config_dict = config
    config_manager.config = ConfigObject(config)
    
    return config_manager.write_to_temp_config()


def write_to_fixed_temp_config(config: Union[Dict[str, Any], ConfigObject], name="temp_config") -> str:
    """
    Write a configuration to a fixed temporary file.
    
    Args:
        config: Configuration dictionary or ConfigObject to write
        name (str): Base name for the temporary file
        
    Returns:
        str: Path to the fixed temporary configuration file
    """
    config_manager = ConfigManager()
    
    # Convert ConfigObject to dict if needed
    if isinstance(config, ConfigObject):
        config = config.to_dict()
        
    # Replace the loaded config with the provided one
    config_manager.config_dict = config
    config_manager.config = ConfigObject(config)
    
    return config_manager.write_to_fixed_temp_config(name)


def get_last_temp_config_path() -> Optional[str]:
    """
    Get the path to the most recently created temporary config file.
    
    Returns:
        str or None: Path to the last temporary config file, or None if none exists
    """
    return ConfigManager.get_last_temp_config_path()
    