# shapes-morphing-from-scratch
This repository explores the use of diffusion models for smoothly morphing one shape into another. Instead of relying on traditional interpolation techniques, we leverage generative diffusion processes to learn continuous transitions between source and target geometries.

# Installation

This project requires **Python3.12**.
We recommend using a virtual environment to keep dependencies isolated.

### 1. Clone the repository
```bash
git clone https://github.com/Stalko123/shapes-morphing-from-scratch.git
cd shapes-morphing-from-scratch
```
### 2. Create virtual environment
On Linux/macOS:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```
On Windows:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```
If you do not have Python3.12 installed on your device, you may want to install it.
On Linux/macOS:
```bash
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL -l
pyenv install 3.12.9
pyenv local 3.12.9
```
Now you can run the Python3.12.9 installation command.
On Windows:
Install Python 3.12 from the Microsoft Store or python.org, then use the commands above.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run demo script
```bash
chmod +x ./*.sh
./scripts/demoscripttodo.sh
```








# References: 
1- Denoising Diffusion Probabilistic Models: https://arxiv.org/pdf/2006.11239

2- Flow matching for generative modeling : https://arxiv.org/pdf/2210.02747

3- Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling : https://arxiv.org/abs/2106.01357

4- Deep Unsupervised Learning using Nonequilibrium Thermodynamics : https://arxiv.org/abs/1503.03585



