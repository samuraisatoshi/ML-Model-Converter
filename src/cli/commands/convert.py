import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.entities.model_info import ModelInfo
from core.entities.conversion_config import ConversionConfig
from core.enums.model_types import ModelType
from services.conversion_service import ConversionService
from services.validation_service import ValidationService
from infrastructure.storage.local_storage import LocalStorageService
from infrastructure.logging.file_logger import CombinedLogger


@click.command("convert")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output path for the converted model"
)
@click.option(
    "--input-shape", "-s",
    type=str,
    default="1,3,224,224",
    help="Input shape as comma-separated values (default: 1,3,224,224)"
)
@click.option(
    "--optimization", "-opt",
    type=click.Choice(["none", "default", "aggressive"]),
    default="default",
    help="Optimization level (default: default)"
)
@click.option(
    "--quantization/--no-quantization",
    default=True,
    help="Enable/disable quantization (default: enabled)"
)
@click.option(
    "--inference-type", "-t",
    type=click.Choice(["FLOAT32", "FLOAT16", "INT8"]),
    default="FLOAT32",
    help="Inference type (default: FLOAT32)"
)
@click.option(
    "--allow-custom-ops",
    is_flag=True,
    help="Allow custom operations"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def convert_command(
    model_path: Path,
    output: Path,
    input_shape: str,
    optimization: str,
    quantization: bool,
    inference_type: str,
    allow_custom_ops: bool,
    verbose: bool
):
    """
    Convert a machine learning model to TensorFlow Lite format.
    
    MODEL_PATH: Path to the input model file
    
    Examples:
    
        # Convert PyTorch model with default settings
        python -m cli.main convert model.pth
        
        # Convert with custom output path and settings
        python -m cli.main convert model.pth -o converted.tflite -s 1,3,256,256
        
        # Convert with quantization disabled
        python -m cli.main convert model.h5 --no-quantization
    """
    try:
        # Parse input shape
        try:
            input_shape_tuple = tuple(map(int, input_shape.split(',')))
        except ValueError:
            click.echo(f"❌ Error: Invalid input shape format: {input_shape}", err=True)
            click.echo("   Expected format: 'batch,channels,height,width' (e.g., '1,3,224,224')")
            sys.exit(1)
        
        # Determine model type from file extension
        try:
            model_type = ModelType.from_extension(str(model_path))
        except ValueError as e:
            click.echo(f"❌ Error: {str(e)}", err=True)
            sys.exit(1)
        
        # Set default output path if not provided
        if output is None:
            output = model_path.parent / f"{model_path.stem}_converted.tflite"
        
        # Initialize services
        log_level = "DEBUG" if verbose else "INFO"
        logger = CombinedLogger(console_level=log_level, file_level="DEBUG")
        storage = LocalStorageService()
        validation = ValidationService(logger)
        conversion_service = ConversionService(storage, logger, validation)
        
        click.echo("🔄 ML Model Converter")
        click.echo("=" * 50)
        click.echo(f"📁 Input:  {model_path}")
        click.echo(f"📁 Output: {output}")
        click.echo(f"🏗️  Type:   {model_type.value}")
        click.echo(f"📐 Shape:  {input_shape_tuple}")
        click.echo(f"⚙️  Config: {optimization} optimization, quantization={'ON' if quantization else 'OFF'}")
        click.echo("=" * 50)
        
        # Create model info
        model_info = ModelInfo(
            file_path=model_path,
            model_type=model_type,
            input_shape=input_shape_tuple,
            model_size=model_path.stat().st_size,
            metadata={}
        )
        
        # Create conversion config
        config = ConversionConfig(
            output_path=output,
            optimization_level=optimization,
            quantization=quantization,
            inference_type=inference_type,
            allow_custom_ops=allow_custom_ops
        )
        
        # Validate inputs
        click.echo("🔍 Validating input model...")
        if not conversion_service.validate_model_info(model_info):
            click.echo("❌ Model validation failed", err=True)
            sys.exit(1)
        
        # Perform conversion
        click.echo("🚀 Starting conversion...")
        
        with click.progressbar(length=100, label="Converting") as bar:
            # Simulate progress updates (in a real implementation, this would be integrated with the converter)
            bar.update(25)
            result = conversion_service.convert_model(model_info, config)
            bar.update(75)
        
        # Display results
        if result.is_successful:
            click.echo("\n✅ Conversion completed successfully!")
            click.echo("=" * 50)
            click.echo(f"📊 Original size:    {result.original_size_mb:.2f} MB")
            click.echo(f"📊 Converted size:   {result.converted_size_mb:.2f} MB")
            click.echo(f"📊 Size reduction:   {result.size_reduction_percent:.1f}%")
            click.echo(f"⏱️  Execution time:   {result.execution_time:.2f} seconds")
            click.echo(f"📁 Output saved to:  {result.output_path}")
            
            # Test the converted model
            if click.confirm("\n🧪 Would you like to test the converted model?"):
                test_converted_model(result.output_path, input_shape_tuple)
        else:
            click.echo(f"\n❌ Conversion failed: {result.error_message}", err=True)
            if verbose and result.error_message:
                click.echo(f"\n🔍 Error details:\n{result.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        click.echo("\n⚠️ Conversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(f"\n🔍 Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def test_converted_model(model_path: Path, input_shape: tuple):
    """Test the converted TFLite model."""
    try:
        import tensorflow as tf
        import numpy as np
        
        click.echo("🧪 Testing converted model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        click.echo(f"📐 Input shape:  {input_details[0]['shape']}")
        click.echo(f"📐 Output shape: {output_details[0]['shape']}")
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        click.echo(f"✅ Model test successful!")
        click.echo(f"📊 Output shape: {output.shape}")
        click.echo(f"📊 Output range: [{output.min():.6f}, {output.max():.6f}]")
        
    except ImportError:
        click.echo("⚠️ TensorFlow not available for testing", err=True)
    except Exception as e:
        click.echo(f"❌ Model test failed: {str(e)}", err=True)