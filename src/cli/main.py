#!/usr/bin/env python3
"""
ML Model Converter CLI
Main command-line interface for the model converter.
"""

import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands.convert import convert_command
from cli.commands.validate import validate_command
from cli.commands.list_formats import list_formats_command


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    ML Model Converter - Convert machine learning models to TensorFlow Lite format.
    
    This tool provides a command-line interface for converting models from various
    frameworks (PyTorch, TensorFlow, Keras, ONNX) to TensorFlow Lite format.
    """
    pass


# Add commands to the CLI group
cli.add_command(convert_command)
cli.add_command(validate_command)
cli.add_command(list_formats_command)


if __name__ == "__main__":
    cli()