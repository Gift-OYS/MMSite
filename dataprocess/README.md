## This tutorial is about how to process data
### 1. Download data
You should download the ProTAD dataset `protad.tsv` from [Link](https://drive.google.com/file/d/1oyl9JVfEvDk72HdtFaPrRKgUDmI_AFHb/view?usp=sharing) and put this file in this folder.

(In order to better reproduce the main experimental results of the paper, we have also uploaded the splitted training dataset, validation dataset, and test dataset at the clustering threshold at 10% in above URL.)
### 2. Run the following codes to generate protad.fasta
```python
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('/path/to/protad.tsv', sep='\t')
data.dropna(subset=['Entry', 'Function', 'Protein names', 'Active site'], inplace=True)
data.reset_index(drop=True, inplace=True)

with open('/path/to/protad.fasta', 'w') as f:
    for idx in tqdm(range(len(data))):
        entry = data.loc[idx, 'Entry']
        seq = data.loc[idx, 'Sequence']
        f.write(f'>{entry}\n{seq}\n')
```
### 3. Run the following scripts to cluster the sequences
Tips: You should install `mmseqs2` first.
```linux
mmseqs createdb /path/to/protad.fasta db
mmseqs cluster db cluster /path/to/tmp_folder --min-seq-id 0.1 --cov-mode 1
mmseqs createtsv db db cluster /path/to/mmseqs2.tsv
rm -rf cluster* /path/to/tmp_folder db*
```
### 4. Run split.py in this folder to split the data
Tips: You should specify the corresponding paths first in split.py
```python
python split.py
```
You can put the generated `train.tsv`, `valid.tsv` and `test.tsv` into `datasets` folder.

We also provide the our processed `train.tsv`, `valid.tsv` and `test.tsv` in above Link. You can download them and put them into `datasets` folder directly.
### 5. Obtain generated descriptions from agent model
Furthermore, as the agent model to generate descriptions during validation and testing, we use the pre-trained Prot2Text<sub>BASE</sub> model. You should follow the instruction below to obtain `generated_desc.json`.
#### 5.1 Download the Pre-trained Model
You can download the pre-trained model of Prot2Text from the link in its official repository.
#### 5.2 Prepare scripts
You should clone the repository of official Prot2Text, specify the corrsponding paths in `agent.py` in this folder and put the `agent.py` to the cloned directory.

Tips: You should also install the required environments of Prot2Text.
#### 5.3 Generate descriptions
You can run the following command to generate descriptions `/path/to/datasets/generated_desc.json`.
```bash
python agent.py
```
