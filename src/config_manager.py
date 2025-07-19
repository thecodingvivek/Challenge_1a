#!/usr/bin/env python3
"""
Configuration Manager for PDF Structure Detection System
Provides flexible configuration management for different deployment scenarios
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    max_pages: int = 50
    time_limit_seconds: float = 10.0
    memory_limit_mb: int = 512
    max_workers: int = 2
    batch_size: int = 5
    use_fast_mode: bool = False
    enable_contextual_analysis: bool = True
    enable_ml_models: bool = True

@dataclass
class ModelConfig:
    """Model configuration"""
    model_path: str = "advanced_heading_classifier.pkl"
    feature_selection_k: int = 50
    ensemble_voting: str = "soft"  # "soft" or "hard"
    confidence_threshold: float = 0.3
    use_class_weights: bool = True
    cross_validation_folds: int = 5

@dataclass
class HeuristicConfig:
    """Heuristic configuration"""
    title_score_threshold: float = 2.0
    h1_score_threshold: float = 2.0
    h2_score_threshold: float = 2.0
    h3_score_threshold: float = 2.0
    font_size_weight: float = 1.0
    position_weight: float = 0.8
    length_weight: float = 0.6
    pattern_weight: float = 1.2
    context_weight: float = 1.5

@dataclass
class OutputConfig:
    """Output configuration"""
    format: str = "json"  # "json", "xml", "csv"
    indent: int = 2
    ensure_ascii: bool = False
    include_confidence: bool = False
    include_metadata: bool = False
    max_outline_items: int = 100
    min_text_length: int = 3

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_performance_logging: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration"""
    processing: ProcessingConfig
    model: ModelConfig
    heuristic: HeuristicConfig
    output: OutputConfig
    logging: LoggingConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary"""
        return cls(
            processing=ProcessingConfig(**data.get('processing', {})),
            model=ModelConfig(**data.get('model', {})),
            heuristic=HeuristicConfig(**data.get('heuristic', {})),
            output=OutputConfig(**data.get('output', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )

class ConfigManager:
    """Configuration manager for the PDF structure detection system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or environment"""
        
        # Default configuration
        config = SystemConfig(
            processing=ProcessingConfig(),
            model=ModelConfig(),
            heuristic=HeuristicConfig(),
            output=OutputConfig(),
            logging=LoggingConfig()
        )
        
        # Load from file if specified
        if self.config_path and os.path.exists(self.config_path):
            try:
                config = self._load_from_file(self.config_path)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        # Override with environment variables
        config = self._override_with_env(config)
        
        return config
    
    def _load_from_file(self, file_path: str) -> SystemConfig:
        """Load configuration from file"""
        
        path = Path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return SystemConfig.from_dict(data)
    
    def _override_with_env(self, config: SystemConfig) -> SystemConfig:
        """Override configuration with environment variables"""
        
        # Processing config
        config.processing.max_pages = int(os.getenv('PDF_MAX_PAGES', config.processing.max_pages))
        config.processing.time_limit_seconds = float(os.getenv('PDF_TIME_LIMIT', config.processing.time_limit_seconds))
        config.processing.memory_limit_mb = int(os.getenv('PDF_MEMORY_LIMIT', config.processing.memory_limit_mb))
        config.processing.max_workers = int(os.getenv('PDF_MAX_WORKERS', config.processing.max_workers))
        config.processing.use_fast_mode = os.getenv('PDF_FAST_MODE', 'false').lower() == 'true'
        
        # Model config
        config.model.model_path = os.getenv('PDF_MODEL_PATH', config.model.model_path)
        config.model.confidence_threshold = float(os.getenv('PDF_CONFIDENCE_THRESHOLD', config.model.confidence_threshold))
        
        # Output config
        config.output.format = os.getenv('PDF_OUTPUT_FORMAT', config.output.format)
        config.output.include_confidence = os.getenv('PDF_INCLUDE_CONFIDENCE', 'false').lower() == 'true'
        config.output.include_metadata = os.getenv('PDF_INCLUDE_METADATA', 'false').lower() == 'true'
        
        # Logging config
        config.logging.level = os.getenv('PDF_LOG_LEVEL', config.logging.level)
        config.logging.file_path = os.getenv('PDF_LOG_FILE', config.logging.file_path)
        
        return config
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        
        log_config = self.config.logging
        
        # Configure logging level
        level = getattr(logging, log_config.level.upper(), logging.INFO)
        
        # Configure handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(log_config.format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # File handler
        if log_config.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size_mb * 1024 * 1024,
                backupCount=log_config.backup_count
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(log_config.format)
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True
        )
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        
        # Update nested configuration
        for section, values in updates.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_config(self, file_path: str, format: str = 'json'):
        """Save configuration to file"""
        
        data = self.config.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() in ['yaml', 'yml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        
        errors = []
        
        # Validate processing config
        if self.config.processing.max_pages <= 0:
            errors.append("max_pages must be positive")
        
        if self.config.processing.time_limit_seconds <= 0:
            errors.append("time_limit_seconds must be positive")
        
        if self.config.processing.memory_limit_mb <= 0:
            errors.append("memory_limit_mb must be positive")
        
        # Validate model config
        if self.config.model.confidence_threshold < 0 or self.config.model.confidence_threshold > 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        # Validate heuristic config
        if self.config.heuristic.title_score_threshold < 0:
            errors.append("title_score_threshold must be non-negative")
        
        # Validate output config
        if self.config.output.format not in ['json', 'xml', 'csv']:
            errors.append("output format must be json, xml, or csv")
        
        if errors:
            logger.error(f"Configuration validation errors: {errors}")
            return False
        
        return True
    
    def get_deployment_config(self, deployment_type: str) -> SystemConfig:
        """Get configuration for specific deployment type"""
        
        config = self.config
        
        if deployment_type == 'development':
            config.logging.level = 'DEBUG'
            config.processing.use_fast_mode = False
            config.output.include_confidence = True
            config.output.include_metadata = True
        
        elif deployment_type == 'production':
            config.logging.level = 'INFO'
            config.processing.use_fast_mode = False
            config.model.confidence_threshold = 0.5
            config.output.include_confidence = False
            config.output.include_metadata = False
        
        elif deployment_type == 'fast':
            config.processing.use_fast_mode = True
            config.processing.time_limit_seconds = 5.0
            config.processing.enable_contextual_analysis = False
            config.processing.enable_ml_models = False
        
        elif deployment_type == 'high_accuracy':
            config.processing.use_fast_mode = False
            config.processing.enable_contextual_analysis = True
            config.processing.enable_ml_models = True
            config.model.confidence_threshold = 0.3
            config.heuristic.context_weight = 2.0
        
        elif deployment_type == 'docker':
            config.processing.max_workers = 2
            config.processing.memory_limit_mb = 512
            config.logging.file_path = None  # Use console only
            config.model.model_path = '/app/models/advanced_heading_classifier.pkl'
        
        return config

# Configuration presets
CONFIG_PRESETS = {
    'hackathon': {
        'processing': {
            'max_pages': 50,
            'time_limit_seconds': 10.0,
            'memory_limit_mb': 512,
            'max_workers': 2,
            'use_fast_mode': False
        },
        'model': {
            'model_path': '/app/models/advanced_heading_classifier.pkl',
            'confidence_threshold': 0.3
        },
        'output': {
            'format': 'json',
            'indent': 2,
            'include_confidence': False
        },
        'logging': {
            'level': 'INFO',
            'file_path': None
        }
    },
    
    'enterprise': {
        'processing': {
            'max_pages': 200,
            'time_limit_seconds': 30.0,
            'memory_limit_mb': 2048,
            'max_workers': 4,
            'use_fast_mode': False,
            'enable_contextual_analysis': True
        },
        'model': {
            'confidence_threshold': 0.4,
            'use_class_weights': True
        },
        'output': {
            'include_confidence': True,
            'include_metadata': True
        },
        'logging': {
            'level': 'INFO',
            'file_path': '/var/log/pdf_processor.log',
            'enable_performance_logging': True
        }
    },
    
    'research': {
        'processing': {
            'time_limit_seconds': 60.0,
            'enable_contextual_analysis': True,
            'enable_ml_models': True
        },
        'model': {
            'confidence_threshold': 0.2,
            'cross_validation_folds': 10
        },
        'output': {
            'include_confidence': True,
            'include_metadata': True
        },
        'logging': {
            'level': 'DEBUG',
            'enable_performance_logging': True
        }
    }
}

def load_preset_config(preset_name: str) -> SystemConfig:
    """Load a preset configuration"""
    
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}")
    
    preset_data = CONFIG_PRESETS[preset_name]
    
    # Start with default config and update with preset
    default_config = SystemConfig(
        processing=ProcessingConfig(),
        model=ModelConfig(),
        heuristic=HeuristicConfig(),
        output=OutputConfig(),
        logging=LoggingConfig()
    )
    
    # Update with preset values
    config_dict = default_config.to_dict()
    
    for section, values in preset_data.items():
        if section in config_dict:
            config_dict[section].update(values)
    
    return SystemConfig.from_dict(config_dict)

def main():
    """Example usage of configuration manager"""
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Load hackathon preset
    hackathon_config = load_preset_config('hackathon')
    
    # Print configuration
    print("Hackathon Configuration:")
    print(json.dumps(hackathon_config.to_dict(), indent=2))
    
    # Save configuration
    config_manager.config = hackathon_config
    config_manager.save_config('hackathon_config.json')
    
    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"Configuration is valid: {is_valid}")

if __name__ == "__main__":
    main()