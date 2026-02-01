#!/bin/bash
# Run ASAM experiments on GTX 3060 12GB
# Save this as run_on_3060.sh and execute on your machine

echo "=============================================="
echo "ASAM 3060 12GB Experiment Runner"
echo "=============================================="

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Create virtual environment
echo ""
echo "Creating Python environment..."
python3 -m venv asam_env
source asam_env/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy seaborn

# Run experiment
echo ""
echo "Running experiments..."
python run_3060_baseline.py

# Show results
echo ""
echo "=============================================="
echo "Experiments completed!"
echo "Results saved in: experiments/results_3060/"
echo "=============================================="
ls -lh experiments/results_3060/

echo ""
echo "To view plots:"
echo "  Open experiments/results_3060/plots_*.png"
