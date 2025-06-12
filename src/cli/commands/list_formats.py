import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.enums.model_types import ModelType
from converters.factory.converter_factory import ConverterFactory
from infrastructure.logging.console_logger import ConsoleLogger


@click.command("list-formats")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed information about each format"
)
@click.option(
    "--model-type", "-t",
    type=click.Choice([mt.value for mt in ModelType]),
    help="Show information for a specific model type only"
)
def list_formats_command(verbose: bool, model_type: str):
    """
    List all supported model formats and converters.
    
    This command displays information about all supported model formats,
    their file extensions, and the available converters.
    
    Examples:
    
        # List all supported formats
        python -m cli.main list-formats
        
        # Show detailed information
        python -m cli.main list-formats --verbose
        
        # Show only PyTorch formats
        python -m cli.main list-formats -t pytorch
    """
    try:
        # Initialize services
        logger = ConsoleLogger(level="ERROR")  # Minimal logging for clean output
        factory = ConverterFactory(logger)
        
        click.echo("🔄 ML Model Converter - Supported Formats")
        click.echo("=" * 60)
        
        # Get available converters
        available_converters = factory.get_available_converters()
        
        if not available_converters:
            click.echo("❌ No converters available")
            return
        
        # Filter by model type if specified
        if model_type:
            target_type = ModelType(model_type)
            available_converters = {
                mt: converter for mt, converter in available_converters.items()
                if mt == target_type
            }
            
            if not available_converters:
                click.echo(f"❌ No converter available for {model_type}")
                return
        
        # Display format information
        for model_type_enum, converter_name in available_converters.items():
            click.echo(f"\n📋 {model_type_enum.value.upper()}")
            click.echo("-" * 30)
            
            # Get supported file formats
            try:
                supported_formats = factory.get_supported_formats(model_type_enum)
                click.echo(f"🏷️  File Extensions: {', '.join(supported_formats)}")
                click.echo(f"🔧 Converter:       {converter_name}")
                
                if verbose:
                    # Get converter instance for more details
                    converter = factory.get_converter(model_type_enum)
                    
                    # Display conversion pipeline
                    click.echo(f"📝 Description:")
                    
                    if model_type_enum == ModelType.PYTORCH:
                        click.echo("   • PyTorch models (.pth, .pt files)")
                        click.echo("   • Supports both state_dict and complete models")
                        click.echo("   • Conversion pipeline: PyTorch → ONNX → TensorFlow → TFLite")
                        click.echo("   • Requires input shape specification")
                        
                    elif model_type_enum == ModelType.TENSORFLOW:
                        click.echo("   • TensorFlow models (.pb files)")
                        click.echo("   • Frozen graph format")
                        click.echo("   • Conversion pipeline: TensorFlow → TFLite")
                        
                    elif model_type_enum == ModelType.KERAS:
                        click.echo("   • Keras models (.h5, .keras files)")
                        click.echo("   • Sequential and Functional API models")
                        click.echo("   • Conversion pipeline: Keras → TFLite")
                        
                    elif model_type_enum == ModelType.ONNX:
                        click.echo("   • ONNX models (.onnx files)")
                        click.echo("   • Cross-platform neural network format")
                        click.echo("   • Conversion pipeline: ONNX → TensorFlow → TFLite")
                    
                    # Show example usage
                    click.echo(f"💡 Example Usage:")
                    example_file = f"model{supported_formats[0]}"
                    click.echo(f"   python -m cli.main convert {example_file}")
                    
                    if model_type_enum == ModelType.PYTORCH:
                        click.echo(f"   python -m cli.main convert {example_file} -s 1,3,224,224")
            
            except Exception as e:
                click.echo(f"❌ Error getting format info: {str(e)}")
        
        # General information
        click.echo(f"\n📊 Summary")
        click.echo("=" * 30)
        click.echo(f"Total supported formats: {len(available_converters)}")
        
        all_extensions = []
        for model_type_enum in available_converters.keys():
            try:
                extensions = factory.get_supported_formats(model_type_enum)
                all_extensions.extend(extensions)
            except:
                pass
        
        click.echo(f"Total file extensions: {len(set(all_extensions))}")
        click.echo(f"Supported extensions: {', '.join(sorted(set(all_extensions)))}")
        
        # Conversion options
        if verbose:
            click.echo(f"\n⚙️ Conversion Options")
            click.echo("=" * 30)
            click.echo("Optimization Levels:")
            click.echo("  • none       - No optimization")
            click.echo("  • default    - Standard optimization (recommended)")
            click.echo("  • aggressive - Maximum optimization")
            click.echo("")
            click.echo("Inference Types:")
            click.echo("  • FLOAT32    - Full precision (largest size)")
            click.echo("  • FLOAT16    - Half precision (smaller size)")
            click.echo("  • INT8       - Integer quantization (smallest size)")
            click.echo("")
            click.echo("Additional Options:")
            click.echo("  • Quantization       - Reduce model size")
            click.echo("  • Custom operations  - Support custom ops")
            click.echo("  • Experimental mode  - Use latest TF features")
        
        # Getting started
        click.echo(f"\n🚀 Getting Started")
        click.echo("=" * 30)
        click.echo("1. Validate your model:")
        click.echo("   python -m cli.main validate your_model.pth")
        click.echo("")
        click.echo("2. Convert your model:")
        click.echo("   python -m cli.main convert your_model.pth")
        click.echo("")
        click.echo("3. For help with a specific command:")
        click.echo("   python -m cli.main convert --help")
        
        if not verbose:
            click.echo(f"\n💡 Use --verbose for detailed information about each format")
            
    except Exception as e:
        click.echo(f"❌ Error listing formats: {str(e)}", err=True)
        sys.exit(1)