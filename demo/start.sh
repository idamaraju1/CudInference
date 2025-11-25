#!/bin/bash

# Startup script for ONNX Text Autocomplete Demo
# This script runs both frontend and backend on port 8000

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ONNX Text Autocomplete Demo Startup ===${NC}\n"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if required files exist
echo "Checking required files..."

if [ ! -f "$PROJECT_ROOT/build/onnx_gpu_engine" ]; then
    echo -e "${RED}ERROR: Engine not found at $PROJECT_ROOT/build/onnx_gpu_engine${NC}"
    echo "Please build the engine first:"
    echo "  cd $PROJECT_ROOT"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. && make -j\$(nproc)"
    exit 1
fi
echo -e "${GREEN}✓${NC} Engine found"

if [ ! -f "$PROJECT_ROOT/model.onnx" ]; then
    echo -e "${RED}ERROR: Model not found at $PROJECT_ROOT/model.onnx${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Model found"

if [ ! -f "$PROJECT_ROOT/tokenizer.json" ]; then
    echo -e "${RED}ERROR: Tokenizer not found at $PROJECT_ROOT/tokenizer.json${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Tokenizer found"

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
fi
echo -e "${GREEN}✓${NC} Python dependencies ready"

# Start the server
echo ""
echo -e "${GREEN}Starting server on port 8000...${NC}"
echo ""
echo -e "Access the demo at: ${GREEN}http://localhost:8000${NC}"
echo -e "Or from network: ${GREEN}http://$(hostname -I | awk '{print $1}'):8000${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

cd "$SCRIPT_DIR"
python3 backend.py --host 0.0.0.0 --port 8000
