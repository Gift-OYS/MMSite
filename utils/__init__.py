def get_max_length(col, config):
    if col == 'Protein names':
        return 48
    if col == 'Organism':
        return 24
    if col == 'Taxonomic lineage':
        return 128
    if col == 'Function':
        return 128
    if col == 'Caution':
        return 72
    if col == 'Miscellaneous':
        return 64
    if col == 'Subunit structure':
        return 72
    if col == 'Induction':
        return 64
    if col == 'Tissue specificity':
        return 64
    if col == 'Developmental stage':
        return 64
    if col == 'Allergenic properties':
        return 48
    if col == 'Biotechnological use':
        return 128
    if col == 'Pharmaceutical use':
        return 64
    if col == 'Involvement in disease':
        return 256
    if col == 'Subcellular location':
        return 64
    if col == 'Post-translational modification':
        return 96
    if col == 'Sequence similarities':
        return 32


def get_str(data, col, idx):
    if col == 'Sequence':
        return False
    if col == 'Protein names':
        return f'The name of protein is {data[col][idx]}.'
    if col == 'Organism':
        organism = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The organism is {organism}."
    if col == 'Taxonomic lineage':
        taxonomic = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The taxonomic lineage of this protein includes {taxonomic}."
    if col == 'Function':
        function = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The function of this protein includes {function}."
    if col == 'Caution':
        caution = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"Caution includes {caution}."
    if col == 'Miscellaneous':
        miscellaneous = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"Some miscellaneous things of this protein includes {miscellaneous}."
    if col == 'Subunit structure':
        subunit = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"Subunit structure of this protein is {subunit}."
    if col == 'Induction':
        induction = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The induction of this protein includes {induction}."
    if col == 'Tissue specificity':
        tissue = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The tissue specificity of this protein is {tissue}."
    if col == 'Developmental stage':
        developmental = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The developmental stage of this protein is {developmental}."
    if col == 'Allergenic properties':
        allergenic = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The allergenic properties includes {allergenic}."
    if col == 'Biotechnological use':
        biotechnologicaluse = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The biotechnological use includes {biotechnologicaluse}."
    if col == 'Pharmaceutical use':
        pharmaceutical_use = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The pharmaceutical use includes {pharmaceutical_use}."
    if col == 'Involvement in disease':
        diseases = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The diseases it could lead involves include {diseases}."
    if col == 'Subcellular location':
        subcellular = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"This protein is usually located in subcellular {subcellular}."
    if col == 'Post-translational modification':
        ptm = data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The post-translational modification of this protein includes {ptm}."
    if col == 'Sequence similarities':
        smilarities= data[col][idx] if type(data[col][idx]) == str else 'unknown'
        return f"The sequence similarities of this protein includes {smilarities}."


def get_col_name_list():
    col_name_list = ['Protein names', 
                    'Organism',
                    'Taxonomic lineage',
                    'Function',
                    'Caution',
                    'Miscellaneous',
                    'Subunit structure',
                    'Induction',
                    'Tissue specificity',
                    'Developmental stage',
                    'Allergenic properties',
                    'Biotechnological use',
                    'Pharmaceutical use',
                    'Involvement in disease',
                    'Subcellular location',
                    'Post-translational modification',
                    'Sequence similarities']
    return col_name_list
