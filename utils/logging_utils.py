#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for configuring logging across the application
"""

import logging
import os

def configure_logging(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s'):
    """Configure global logging settings."""
    logging.basicConfig(level=getattr(logging, level), format=format)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    
    # Configure library-specific loggers
    configure_litellm_logging()
    configure_dspy_logging()

def configure_litellm_logging(level="ERROR"):
    """Configure LiteLLM logging to reduce verbosity."""
    # Set environment variable to disable LiteLLM debug logs
    os.environ["LITELLM_LOG_LEVEL"] = level
    
    # Configure the LiteLLM logger
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(getattr(logging, level))
    
    # Also configure any related loggers
    logging.getLogger("litellm").setLevel(getattr(logging, level))
    
    # Configure httpx logger (used by LiteLLM)
    logging.getLogger("httpx").setLevel(getattr(logging, level))
    
    # Configure urllib3 logger (used by requests)
    logging.getLogger("urllib3").setLevel(getattr(logging, level))

def configure_dspy_logging(level="INFO"):
    """Configure DSPy logging."""
    # Configure the DSPy logger
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.setLevel(getattr(logging, level))
    
    # Also set the OpenAI logger level
    logging.getLogger("openai").setLevel(getattr(logging, level))

def disable_all_litellm_logs():
    """Completely disable all LiteLLM logs."""
    # Disable by setting to CRITICAL level
    configure_litellm_logging("CRITICAL")
    
    # Set environment variables
    os.environ["LITELLM_LOG_LEVEL"] = "CRITICAL"
    os.environ["LITELLM_VERBOSE"] = "False" 