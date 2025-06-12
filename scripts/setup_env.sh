#!/bin/bash
set -e

echo "🚀 Configurando ambiente ML Model Converter..."

# Verificar Python 3.11
python3.11 --version || {
    echo "❌ Python 3.11 não encontrado. Instale primeiro."
    exit 1
}

# Criar ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3.11 -m venv venv
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Instalar dependências
echo "📚 Instalando dependências base..."
pip install -r requirements/base.txt

echo "🌐 Instalando dependências web..."
pip install -r requirements/web.txt

# Criar diretórios necessários
echo "📁 Criando estrutura de diretórios..."
mkdir -p outputs/{converted,temp,logs}
mkdir -p src/{core/{interfaces,entities,exceptions,enums},converters/{base,pytorch,tensorflow,onnx,factory},services,infrastructure/{storage,logging,config},web/{components,pages,utils},cli/{commands,utils}}
mkdir -p tests/{unit/{test_converters,test_services,test_web},integration,fixtures/sample_models}
mkdir -p docs/{api,architecture,user-guide,developer-guide}
mkdir -p config scripts

# Criar arquivos __init__.py
find src tests -type d -exec touch {}/__init__.py \;

echo "✅ Ambiente configurado com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. Para ativar o ambiente: source venv/bin/activate"
echo "2. Para executar a interface web: streamlit run src/web/app.py"
echo "3. Para ver comandos CLI: python -m src.cli.main --help"
echo ""
echo "📚 Documentação disponível em:"
echo "- Guia de desenvolvimento: CLAUDE.md"
echo "- Documentação do usuário: README.md"
