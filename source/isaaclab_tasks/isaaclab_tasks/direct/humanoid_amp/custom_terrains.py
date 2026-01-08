import numpy as np
import torch
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

def slope_terrain(difficulty: float, cfg: TerrainGeneratorCfg.SubTerrainConfig) -> tuple[np.ndarray, float]:
    """
    Generates a flat, sloped terrain.
    
    Args:
        difficulty (float): A value between 0 and 1. We use this to scale the slope angle.
        cfg (SubTerrainConfig): Configuration dictionary containing 'slope_angle_deg'.
        
    Returns:
        tuple[np.ndarray, float]: The heightfield and the recommended spawn height.
    """
    # 1. Parse configuration
    # We define 'slope_angle_deg' in the config. difficulty acts as a multiplier if desired.
    # Default to 0 degrees if not found.
    target_angle_deg = cfg.args.get("slope_angle_deg", 0.0) * difficulty
    
    # 2. Get dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    # 3. Create the slope
    # We create a linear gradient along the X-axis
    x = np.linspace(0, cfg.size[0], width_pixels)
    slope_rad = np.radians(target_angle_deg)
    
    # Height = x * tan(theta)
    # This creates a ramp going up along the X axis
    height_profile = x * np.tan(slope_rad)
    
    # 4. Expand to 2D grid (repeat profile along Y axis)
    terrain_height = np.tile(height_profile[:, np.newaxis], (1, length_pixels))
    
    # Convert metric height to integer units for the heightfield
    terrain_height_int = (terrain_height / cfg.vertical_scale).astype(np.int16)
    
    # Return heightfield and a safe spawn height (usually slightly above the start)
    return terrain_height_int, 0.1


def noisy_terrain(difficulty: float, cfg: TerrainGeneratorCfg.SubTerrainConfig) -> tuple[np.ndarray, float]:
    """
    Generates terrain from a provided heightfield array (or random noise) scaled by an amplitude.
    
    Args:
        difficulty (float): Scales the noise amplitude (0=flat, 1=max noise).
        cfg (SubTerrainConfig): Configuration containing 'base_heightfield' and 'amplitude_scale'.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # 1. Get the base noise
    # If a specific array isn't provided, we generate random noise
    # In a real scenario, you might load a file here.
    if "base_heightfield" in cfg.args:
        base_noise = np.array(cfg.args["base_heightfield"])
        # Resize to match current dimensions if necessary (simple resizing for demo)
        if base_noise.shape != (width_pixels, length_pixels):
             base_noise = np.resize(base_noise, (width_pixels, length_pixels))
    else:
        # Generate random uniform noise between -1 and 1
        base_noise = np.random.uniform(-1.0, 1.0, (width_pixels, length_pixels))

    # 2. Apply scaling
    # amplitude_scale defines the maximum height variation in meters
    max_amp = cfg.args.get("amplitude_scale", 1.0)
    current_amp = max_amp * difficulty # Scale by difficulty (curriculum)
    
    terrain_height = base_noise * current_amp

    # 3. Convert to integer units
    terrain_height_int = (terrain_height / cfg.vertical_scale).astype(np.int16)

    return terrain_height_int, 0.1