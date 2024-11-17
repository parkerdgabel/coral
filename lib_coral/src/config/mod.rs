use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub chunk_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileConfig {
    pub default_tile_size: Vec<usize>,
    pub max_tile_dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    pub username: String,
    pub email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub user: UserConfig,
    pub tile: TileConfig,
    pub chunk: ChunkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TileStrategy {
    Fixed(Vec<usize>),
    Ratio(f32),
    Equal(usize),
    Tensor, // Use the tensor's shape as the tile size
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkStrategy {
    Single,
    Fixed(usize),
    Adaptive {
        initial_size: usize,
        max_size: usize,
    },
}

impl TileConfig {
    pub fn with_strategy(strategy: TileStrategy) -> Self {
        match strategy {
            TileStrategy::Fixed(sizes) => Self {
                default_tile_size: sizes,
                max_tile_dimensions: 4,
            },
            TileStrategy::Ratio(ratio) => Self {
                default_tile_size: vec![((64.0 * ratio) as usize); 2],
                max_tile_dimensions: 4,
            },
            TileStrategy::Equal(size) => Self {
                default_tile_size: vec![size; 2],
                max_tile_dimensions: 4,
            },
            TileStrategy::Tensor => Self {
                default_tile_size: vec![0; 2],
                max_tile_dimensions: 4,
            },
        }
    }
}

impl ChunkConfig {
    pub fn with_strategy(strategy: ChunkStrategy) -> Self {
        match strategy {
            ChunkStrategy::Single => Self {
                chunk_size: usize::MAX,
            },
            ChunkStrategy::Fixed(size) => Self { chunk_size: size },
            ChunkStrategy::Adaptive {
                initial_size,
                max_size,
            } => Self {
                chunk_size: initial_size.min(max_size),
            },
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            user: UserConfig {
                username: String::from("unknown"),
                email: String::from("unknown@example.com"),
            },
            tile: TileConfig {
                default_tile_size: vec![64, 64],
                max_tile_dimensions: 4,
            },
            chunk: ChunkConfig {
                chunk_size: 1024 * 1024,
            },
        }
    }
}
impl Config {
    pub fn load() -> Result<Self> {
        // First try the current directory and its parents
        let current_dir = std::env::current_dir()?;
        if let Ok(config) = Self::find_in_parents(&current_dir) {
            return Ok(config);
        }

        // Then try the home directory
        if let Some(home_dir) = dirs::home_dir() {
            if let Ok(config) = Self::load_from_dir(&home_dir) {
                return Ok(config);
            }
        }

        // If no config is found, return default
        Ok(Self::default())
    }

    fn find_in_parents(start_dir: &Path) -> Result<Self> {
        let mut current = start_dir.to_path_buf();
        while let Some(parent) = current.parent() {
            if let Ok(config) = Self::load_from_dir(&current) {
                return Ok(config);
            }
            current = parent.to_path_buf();
        }
        anyhow::bail!("No config found in parent directories")
    }

    fn load_from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join(".coral").join("config.toml");
        if !config_path.exists() {
            anyhow::bail!("Config file not found at {:?}", config_path);
        }

        let contents = std::fs::read_to_string(config_path)?;
        let mut config: Config = toml::from_str(&contents)?;

        // Apply tile strategy if specified in config
        if let Some(tile_strategy) = Self::get_tile_strategy_from_env() {
            config.tile = TileConfig::with_strategy(tile_strategy);
        }

        // Apply chunk strategy if specified in config
        if let Some(chunk_strategy) = Self::get_chunk_strategy_from_env() {
            config.chunk = ChunkConfig::with_strategy(chunk_strategy);
        }

        Ok(config)
    }

    fn get_tile_strategy_from_env() -> Option<TileStrategy> {
        std::env::var("CORAL_TILE_STRATEGY")
            .ok()
            .and_then(|s| match s.as_str() {
                "tensor" => Some(TileStrategy::Tensor),
                _ if s.starts_with("fixed:") => {
                    let sizes: Vec<usize> = s
                        .split(':')
                        .nth(1)?
                        .split(',')
                        .filter_map(|n| n.parse().ok())
                        .collect();
                    Some(TileStrategy::Fixed(sizes))
                }
                _ if s.starts_with("ratio:") => {
                    s.split(':').nth(1)?.parse().ok().map(TileStrategy::Ratio)
                }
                _ if s.starts_with("equal:") => {
                    s.split(':').nth(1)?.parse().ok().map(TileStrategy::Equal)
                }
                _ => None,
            })
    }

    fn get_chunk_strategy_from_env() -> Option<ChunkStrategy> {
        std::env::var("CORAL_CHUNK_STRATEGY")
            .ok()
            .and_then(|s| match s.as_str() {
                "single" => Some(ChunkStrategy::Single),
                _ if s.starts_with("fixed:") => {
                    s.split(':').nth(1)?.parse().ok().map(ChunkStrategy::Fixed)
                }
                _ if s.starts_with("adaptive:") => {
                    let parts: Vec<&str> = s.split(':').collect();
                    if parts.len() == 3 {
                        let initial = parts[1].parse().ok()?;
                        let max = parts[2].parse().ok()?;
                        Some(ChunkStrategy::Adaptive {
                            initial_size: initial,
                            max_size: max,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let config_dir = path.join(".coral");
        std::fs::create_dir_all(&config_dir)?;

        let config_path = config_dir.join("config.toml");
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(config_path, contents)?;

        Ok(())
    }
}
