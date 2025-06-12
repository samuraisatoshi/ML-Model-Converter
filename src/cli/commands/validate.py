import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.entities.model_info import ModelInfo
from core.enums.model_types import ModelType
from services.validation_service import ValidationService
from infrastructure.logging.console_logger import ConsoleLogger


@click.command("validate")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--input-shape", "-s",
    type=str,
    default="1,3,224,224",
    help="Input shape as comma-separated values (default: 1,3,224,224)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--check-integrity",
    is_flag=True,
    help="Perform deep integrity check of the model file"
)
def validate_command(
    model_path: Path,
    input_shape: str,
    verbose: bool,
    check_integrity: bool
):
    """
    Validate a machine learning model file.
    
    MODEL_PATH: Path to the model file to validate
    
    This command checks if a model file is valid and can be converted.
    It performs various validation checks including file existence,
    format validation, and optionally deep integrity checks.
    
    Examples:
    
        # Basic validation
        python -m cli.main validate model.pth
        
        # Validation with custom input shape
        python -m cli.main validate model.h5 -s 1,3,256,256
        
        # Deep integrity check
        python -m cli.main validate model.onnx --check-integrity
    """
    try:
        # Parse input shape
        try:
            input_shape_tuple = tuple(map(int, input_shape.split(',')))
        except ValueError:
            click.echo(f"❌ Error: Invalid input shape format: {input_shape}", err=True)
            click.echo("   Expected format: 'batch,channels,height,width' (e.g., '1,3,224,224')")
            sys.exit(1)
        
        # Initialize validation service
        log_level = "DEBUG" if verbose else "INFO"
        logger = ConsoleLogger(level=log_level)
        validation_service = ValidationService(logger)
        
        click.echo("🔍 ML Model Validator")
        click.echo("=" * 50)
        click.echo(f"📁 Model:  {model_path}")
        click.echo(f"📐 Shape:  {input_shape_tuple}")
        click.echo("=" * 50)
        
        # Step 1: Basic file validation
        click.echo("1️⃣ Checking file existence and accessibility...")
        file_valid = validation_service.validate_model_file(str(model_path))
        
        if file_valid:
            click.echo("   ✅ File validation passed")
        else:
            click.echo("   ❌ File validation failed", err=True)
            sys.exit(1)
        
        # Step 2: Determine and validate model type
        click.echo("2️⃣ Detecting model type...")
        try:
            model_type = ModelType.from_extension(str(model_path))
            click.echo(f"   ✅ Model type detected: {model_type.value}")
        except ValueError as e:
            click.echo(f"   ❌ {str(e)}", err=True)
            sys.exit(1)
        
        # Step 3: Create model info and validate format
        click.echo("3️⃣ Validating model format...")
        try:
            model_info = ModelInfo(
                file_path=model_path,
                model_type=model_type,
                input_shape=input_shape_tuple,
                model_size=model_path.stat().st_size,
                metadata={}
            )
            
            format_valid = validation_service.validate_model_format(model_info)
            
            if format_valid:
                click.echo("   ✅ Model format validation passed")
            else:
                click.echo("   ❌ Model format validation failed", err=True)
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"   ❌ Error creating model info: {str(e)}", err=True)
            sys.exit(1)
        
        # Step 4: Deep integrity check (optional)
        if check_integrity:
            click.echo("4️⃣ Performing deep integrity check...")
            try:
                integrity_valid = validation_service.validate_model_integrity(model_info)
                
                if integrity_valid:
                    click.echo("   ✅ Model integrity check passed")
                else:
                    click.echo("   ❌ Model integrity check failed", err=True)
                    click.echo("   ⚠️ The model file may be corrupted or in an unexpected format")
                    sys.exit(1)
                    
            except Exception as e:
                click.echo(f"   ❌ Integrity check error: {str(e)}", err=True)
                if verbose:
                    import traceback
                    click.echo(f"   🔍 Details: {traceback.format_exc()}")
                sys.exit(1)
        
        # Display model information
        click.echo("\n📊 Model Information")
        click.echo("=" * 50)
        click.echo(f"Filename:     {model_info.filename}")
        click.echo(f"Type:         {model_info.model_type.value}")
        click.echo(f"Size:         {model_info.size_mb:.2f} MB")
        click.echo(f"Extension:    {model_info.extension}")
        click.echo(f"Input Shape:  {model_info.input_shape}")
        
        if verbose and model_info.metadata:
            click.echo(f"Metadata:     {model_info.metadata}")
        
        # Check conversion compatibility
        click.echo("\n🔄 Conversion Compatibility")
        click.echo("=" * 30)
        
        # Import here to avoid circular imports
        from converters.factory.converter_factory import ConverterFactory
        
        factory = ConverterFactory(logger)
        
        if factory.is_supported(model_type):
            converter = factory.get_converter(model_type)
            can_convert = converter.can_convert(model_info)
            
            if can_convert:
                click.echo("✅ This model can be converted to TensorFlow Lite")
                
                # Show supported formats
                supported_formats = converter.get_supported_formats()
                click.echo(f"📋 Supported formats: {', '.join(supported_formats)}")
                
                # Suggest conversion command
                click.echo(f"\n💡 To convert this model, run:")
                click.echo(f"   python -m cli.main convert {model_path}")
                
            else:
                click.echo("❌ This model cannot be converted with the current converter")
                click.echo("   Please check the model format or provide additional metadata")
        else:
            click.echo(f"❌ No converter available for {model_type.value} models")
            click.echo("   This model type is not yet supported")
        
        # Validation summary
        click.echo("\n✅ Validation Summary")
        click.echo("=" * 30)
        click.echo("All validation checks passed!")
        
        if not check_integrity:
            click.echo("\n💡 For a more thorough check, use --check-integrity")
        
    except KeyboardInterrupt:
        click.echo("\n⚠️ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error during validation: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(f"\n🔍 Traceback:\n{traceback.format_exc()}")
        sys.exit(1)