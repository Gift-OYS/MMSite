import os
import random
from tqdm import tqdm
import pandas as pd
import sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)

from utils import get_col_name_list

data = pd.read_csv('/path/to/protad.tsv', sep='\t')

def read_clusters():
    cluster_to_proteins = {}
    split_file = '/path/to/mmseqs2.tsv'
    with open(split_file) as f:
        for line in tqdm(f):
            cluster_id, protein_id = line.strip().split("\t")
            if cluster_id not in cluster_to_proteins:
                cluster_to_proteins[cluster_id] = []
            cluster_to_proteins[cluster_id].append(protein_id)
    cluster_ids = list(cluster_to_proteins.keys())
    random.shuffle(cluster_ids)
    return cluster_to_proteins, cluster_ids


def select_k_from_clusters(clusters, k=1):
    selected_elements = set()
    all_elements = []
    for elements in tqdm(clusters.values()):
        all_elements.extend(elements)
    for elements in tqdm(clusters.values()):
        selected = random.sample(elements, min(k, len(elements)))
        selected_elements.update(selected)
        all_elements = [x for x in all_elements if x not in selected]
    remaining_selections = k - len(selected_elements)
    if remaining_selections > 0 and len(all_elements) > 0:
        remaining_selected = random.sample(all_elements, min(remaining_selections, len(all_elements)))
        selected_elements.update(remaining_selected)
    return selected_elements


def get_data():
    cluster_to_proteins, cluster_ids = read_clusters()
    
    train_val_split_point = int(len(cluster_ids) * 0.8)
    val_test_split_point = int(len(cluster_ids) * 0.9)
    train_cluster_ids = set(cluster_ids[:train_val_split_point])
    val_cluster_ids = set(cluster_ids[train_val_split_point:val_test_split_point])
    test_cluster_ids = set(cluster_ids[val_test_split_point:])
    
    train_cluster = {k: v for k, v in cluster_to_proteins.items() if k in train_cluster_ids}
    val_cluster = {k: v for k, v in cluster_to_proteins.items() if k in val_cluster_ids}
    test_cluster = {k: v for k, v in cluster_to_proteins.items() if k in test_cluster_ids}
    
    train_proteins = select_k_from_clusters(train_cluster)
    val_proteins = select_k_from_clusters(val_cluster)
    test_proteins = select_k_from_clusters(test_cluster)

    return list(train_proteins), list(val_proteins), list(test_proteins)


if __name__ == '__main__':
    train_proteins, val_proteins, test_proteins = get_data()
    train = data[data['Entry'].isin(train_proteins)]
    val = data[data['Entry'].isin(val_proteins)]
    test = data[data['Entry'].isin(test_proteins)]
    
    print(f"Writting ...")
    train.to_csv('/path/to/datasets/train.tsv', sep='\t', index=False)
    val.to_csv('/path/to/datasets/valid.tsv', sep='\t', index=False)
    test.to_csv('/path/to/datasets/test.tsv', sep='\t', index=False)
    print(f"Done!")
