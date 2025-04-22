## Getting Started
## Installation

1. Install Linux prereqs
    - Conda
    Development tools (gcc, etc)
    - Ubuntu (sudo apt install build-essentials)
    - Fedora (sudo dnf install sudo dnf install @development-tools)
    - Terraform (https://developer.hashicorp.com/terraform/install)

2. Clone the repo  
   ```bash
   git clone https://github.com/doodmeister/human-ai-cognition.git
   cd human-ai-cognition
   ```

3. Setup the environment
```bash
# Create a conda environment with a specific Python version (e.g., Python 3.8)
conda create -n human-ai-cognition python=3.9 -y

# Activate the conda environment
conda activate human-ai-cognition

# Install dependencies
pip install -r requirements.txt
