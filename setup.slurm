#!/bin/bash

# ============================================================================
# Setup Script for SikuBERT SLURM Training
# ============================================================================

echo "=========================================="
echo "SikuBERT SLURM Training Setup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p models
mkdir -p data
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "${GREEN}✓ Python version is compatible${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Python 3.11+ required (for scikit-learn>=1.8.0)${NC}"
fi
echo ""

# Check if pip is available
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip3 is available${NC}"
else
    echo -e "${YELLOW}⚠ Warning: pip3 not found${NC}"
fi
echo ""

# Offer to install dependencies
echo "Do you want to install Python dependencies now? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo "Skipping dependency installation"
    echo "You can install them later with: pip3 install -r requirements.txt"
fi
echo ""

# Check SLURM availability
echo "Checking SLURM availability..."
if command -v sbatch &> /dev/null; then
    echo -e "${GREEN}✓ SLURM is available${NC}"
    
    # Show available partitions
    echo ""
    echo "Available SLURM partitions:"
    sinfo -s
else
    echo -e "${YELLOW}⚠ Warning: SLURM commands not found${NC}"
    echo "This might be normal if you're setting up on a login node"
fi
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA is available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
else
    echo -e "${YELLOW}⚠ Warning: CUDA not detected on this node${NC}"
    echo "This is normal if you're on a login node"
fi
echo ""

# Configuration reminder
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Edit config.slurm with your data paths and parameters"
echo "   nano config.slurm"
echo ""
echo "2. Update email in SLURM scripts"
echo "   nano run.slurm"
echo "   # Change: #SBATCH --mail-user=your.email@example.com"
echo ""
echo "3. Adjust SLURM resources in the script headers:"
echo "   - Partition name (#SBATCH --partition=...)"
echo "   - Number of GPUs (#SBATCH --gres=gpu:...)"
echo "   - Memory allocation (#SBATCH --mem=...)"
echo "   - Time limit (#SBATCH --time=...)"
echo ""
echo "4. Submit your job:"
echo "   sbatch run.slurm"
echo ""
echo "5. Monitor your job:"
echo "   ./slurm_helper.slurm status"
echo "   ./slurm_helper.slurm tail <job_id>"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
