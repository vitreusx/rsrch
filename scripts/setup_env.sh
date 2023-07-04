conda create -p ./env --file conda-linux-64.lock
conda activate ./env
pip install -r requirements.txt
poetry install
