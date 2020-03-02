# NAS-CD

Neural Architecture Search for Conditional Dependences

## Install

Install open-mpi with [Homebrew](https://docs.brew.sh/Installation):

```bash
brew install open-mpi
```

Create a conda virtual environment:

```bash
conda create -n dh python=3.7
```

Once created, activate it:

```bash
conda activate dh
```

Now, install DeepHyper from source to access develop branches:

```bash
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
git checkout feature/pblp
pip install -e .
```

Now install `nascd`, by first cloning the repo:

```bash
cd ..
git clone https://github.com/Deathn0t/NAS-CD.git
cd NAS-CD/
pip install -e .
```

Try the installation by running:

```bash
cd nascd/xor/
python problem.py
```

The expected output is the following:

```bash
Using TensorFlow backend.
Problem is:
 * SEED = 2019 *
    - search space   : nascd.xor.search_space.create_search_space
    - data loading   : nascd.xor.load_data.load_data
    - preprocessing  : deephyper.search.nas.model.preprocessing.minmaxstdscaler
    - hyperparameters:
        * verbose: 1
        * batch_size: 32
        * learning_rate: 0.01
        * optimizer: adam
        * num_epochs: 20
        * callbacks: {'EarlyStopping': {'monitor': 'val_r2', 'mode': 'max', 'verbose': 0, 'patience': 5}}
    - loss           : mse
    - metrics        :
        * r2
    - objective      : val_r2__last
    - post-training  : None
```

## Example

Execute an asynchronous model-based search with:

```bash
python -m deephyper.search.nas.ambs --evaluator subprocess --problem nascd.xorandor.problem.Problem --max-evals 64
```

Execute a nascd search with a quick evaluation function `(sum(arch_seq))`:

```bash
python -m nascd.search.rtg_pg --problem nascd.xorandor.problem.Problem --run deephyper.search.nas.model.run.quick.run
```

Parse the logs to create a json file with collected data:

```bash
deephyper-analytics parse deephyper.log
```

Create a jupyter notebook to plot from these data:

```bash
deephyper-analytics single -p $json_data_file
```
