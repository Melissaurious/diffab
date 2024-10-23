import os
import random
import logging
import datetime
import pandas as pd
import joblib
import lmdb
import pickle
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm
# from ..utils.protein import parsers, constants
# from ._base import register_dataset
import sys
sys.path.append('/ibex/user/rioszemm/NanobodiesProject/diffab')


from diffab.utils.protein import parsers, constants
from diffab.datasets._base import register_dataset
from torchvision.transforms import Compose
import json


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    #Added to check if the chains described in the tsv file are indeed there and avoid future errors
    structure = parser.get_structure(entry['pdbcode'], pdb_path)[0]

    # Check if heavy chains are present in the PDB structure
    # if entry['H_chain'] is not None:
    #     if 'H_chain' not in structure:
    #         logging.warning(f'Heavy chain not found for {entry["id"]}  when it is defined. Skipping entry.')
    #         return None

    # Check if heavy chain is present in the PDB structure
    if entry['H_chain'] is not None:
        heavy_chain_present = entry['H_chain'] in [chain.id for chain in structure.get_chains()]
        if not heavy_chain_present:
            logging.warning(f'Heavy chain {entry["H_chain"]} not found for {entry["id"]}. Skipping entry.')
            return None



    # # Check if antigen chains are present in the PDB structure
    # antigen_chains_present = all(c in structure for c in entry['ag_chains'])
    # if not antigen_chains_present:
    #     logging.warning(f'Antigen chains not found for {entry["id"]}  when it is defined. Skipping entry.')
    #     return None

    # Extract all chain IDs from the structure
    structure_chain_ids = [chain.id for chain in structure.get_chains()]

    # Check if all antigen chains are present in the PDB structure
    antigen_chains_present = all(c in structure_chain_ids for c in entry['ag_chains'])
    if not antigen_chains_present:
        missing_chains = [c for c in entry['ag_chains'] if c not in structure_chain_ids]
        logging.warning(f'Antigen chains {missing_chains} not found for {entry["id"]}. Skipping entry.')
        return None
    
    # Check if heavy chains are present in the PDB structure
    # if entry['L_chain'] is not None:
    #     if 'L_chain' not in structure:
    #         logging.warning(f'Light chain not found for {entry["id"]} when it is defined. Skipping entry.')
    #         return None

    # Check if heavy chain is present in the PDB structure
    if entry['L_chain'] is not None:
        heavy_chain_present = entry['L_chain'] in [chain.id for chain in structure.get_chains()]
        if not heavy_chain_present:
            logging.warning(f'Light chain {entry["L_chain"]} not found for {entry["id"]}. Skipping entry.')
            return None 

    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if entry['H_chain'] is not None:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],
                max_resseq = 250    # Chothia, end of Heavy chain Fv
                #original was 113
            ))
        
        if entry['L_chain'] is not None:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['L_chain']],
                max_resseq = 107    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')
    
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None

    # print(parsed)
    return parsed



class SAbDabDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB
    def __init__(
        self, 
        summary_path,  
        chothia_dir,
        json_file,   
        processed_dir,  
        split,        
        split_seed,   
        transform,     
        reset,
        special_filter,
        fold
    ):
        super().__init__()

        self.fold = fold 

        print("conducting fold = ", self.fold)
        self.special_filter = special_filter
        self.summary_path = summary_path
        self.chothia_dir = chothia_dir
        self.split=split
        self.split_seed=split_seed
        self.json_file = json_file

        with open(self.json_file,'r') as f:
            data = f.read().strip().split('\n')
        
        ids = []
        for entry in data:
            try:
                entry_dict = json.loads(entry)
                id_ = entry_dict["entry_id"]
                ids.append(id_)
            except json.JSONDecodeError as e:
                ids.append(entry_dict["entry_id"])

        if self.split == 'train':
            # populate with pdbcode
            self.all_ = self.train_entries = ids
        elif self.split == 'val':
            self.all_ = self.valid_entries = ids


        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        
        # Decide which entries to process based on the split
        if self.split == 'train':
            self.entries_to_process = self.train_entries
        elif self.split == 'val':
            self.entries_to_process = self.valid_entries
        # elif split == 'test':
        #     self.entries_to_process = self.test_entries
        else:
            raise ValueError("Invalid split specified")

        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.reset = reset
    
        self.sabdab_entries = None
        self._load_sabdab_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

        # self.clusters = None
        # self.id_to_cluster = None
        # self._load_clusters(reset)  #comment it and its functions if performing the special_filter = True

        self.ids_in_split = None
        self._load_split()

        self.transform = transform  # You can provide a transformation function
    

        print("self.sabdab_entries",len(self.sabdab_entries),chothia_dir)
        # print(len(self.sabdab_entries))

    def _load_sabdab_entries(self):
        print("performing_load_sabdab_entries")
        df = pd.read_csv(self.summary_path, sep='\t')
        # print("df", len(df))
        
        if not self.special_filter:
            entries_all = []
            for i, row in tqdm(
                df.iterrows(), 
                dynamic_ncols=True, 
                desc='Loading entries',
                total=len(df),
            ):
                entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                    pdbcode = row['pdb'],
                    H = nan_to_empty_string(row['Hchain']),
                    L = nan_to_empty_string(row['Lchain']),
                    Ag = ''.join(split_sabdab_delimited_str(
                        nan_to_empty_string(row['antigen_chain'])
                    ))
                )
                ag_chains = split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                )
                resolution = parse_sabdab_resolution(row['resolution'])
                entry = {
                    'id': entry_id,
                    'pdbcode': row['pdb'],
                    'H_chain': nan_to_none(row['Hchain']),
                    'L_chain': nan_to_none(row['Lchain']),
                    'ag_chains': ag_chains,
                    'ag_type': nan_to_none(row['antigen_type']),
                    'ag_name': nan_to_none(row['antigen_name']),
                    'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                    'resolution': resolution,
                    'method': row['method'],
                    'scfv': row['scfv'],
                }

                if (
                    (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                    and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
                ):
                    entries_all.append(entry)

        else:
            # I already processed these entries, with the information needed, requires resolution and so on
            with open(self.json_file,'r') as f:
                data = f.read().strip().split('\n')
            
            entries_all = []
            for entry in data:
                try:
                    entry_dict = json.loads(entry)
                    #{"pdb": "8cyc", "heavy_chain": "D", "light_chain": "", "antigen_chains": ["A"], "antigen_type": ["protein"], "entry_id": "8cyc_D_X_A", "cluster": "8cyc"}
                    #check if the file is present before appending its index and continue with the rest of the process

                    old_pdb_path = entry_dict['pdb_data_path']
                    file_name = os.path.basename(old_pdb_path)
                    pdb_path = os.path.join(self.chothia_dir,file_name)
                    if not os.path.exists(pdb_path):
                        continue

                    entry_dict["id"] = entry_dict["entry_id"]
                    entry_dict["pdbcode"] = entry_dict["id"]
                    entry_dict["H_chain"] = entry_dict["heavy_chain"]
                    if entry_dict["light_chain"]:
                        entry_dict["L_chain"] = entry_dict["light_chain"]
                    else:
                        entry_dict["L_chain"] = None
                    entry_dict["ag_chains"] = entry_dict["antigen_chains"]
                    entries_all.append(entry_dict)
                except json.JSONDecodeError as e:

                    old_pdb_path = entry['pdb_data_path']
                    file_name = os.path.basename(old_pdb_path)
                    pdb_path = os.path.join(self.chothia_dir,file_name)
                    if not os.path.exists(pdb_path):
                        continue

                    entry["id"] = entry["entry_id"]
                    entry["pdbcode"] = entry["id"]
                    entry["H_chain"] = entry["heavy_chain"]
                    if entry["light_chain"]:
                        entry["L_chain"] = entry["light_chain"]
                    else:
                        entry["L_chain"] = None
                    entry["ag_chains"] = entry["antigen_chains"]
                    entries_all.append(entry)



                # We want only entries from the train, test and validation lists 
        # print("entries_all", len(entries_all))
        # print("entries_all[0] id",entries_all[0]['id'])
        # print("entries_all[0] pdbcocde",entries_all[0]['pdbcode'])
        self.sabdab_entries = entries_all
        # print(self.sabdab_entries[0])
        # print(len(self.sabdab_entries))


        processed_entries = []
        for entry in self.sabdab_entries:
            processed_entries.append(entry)


    def _load_structures(self, reset):
        # if os.path.exists(self._structure_cache_path) and not reset:
        if not os.path.exists(self._structure_cache_path):
            # LMDB database does not exist, so preprocess structures
            self._preprocess_structures()
        
        try:
            with open(self._structure_cache_path + '-ids', 'rb') as f:
                self.db_ids = pickle.load(f)

            if self.db_ids is None:
                raise ValueError("Failed to load db_ids from file")

            # print(self.db_ids)
            self.sabdab_entries = list(
                filter(
                    lambda e: e['id'] in self.db_ids,
                    self.sabdab_entries
                )
            )
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")



    # def _load_structures(self, reset):
    #     if not os.path.exists(self._structure_cache_path) or reset:
    #         if os.path.exists(self._structure_cache_path):
    #             os.unlink(self._structure_cache_path)
    #         self._preprocess_structures()

    #     with open(self._structure_cache_path + '-ids', 'rb') as f:
    #         self.db_ids = pickle.load(f)
    #         # print("self.db_ids", self.db_ids)
    #         # ['6qgy_B__A', '6qgy_B__A', '4c58_B__A', '4c58_B__A', '7rmh_N__AB', '7rmh_N__AB', '6qgy_D__C', '6qgy_D__C']
    #     self.sabdab_entries = list(
    #         filter(
    #             lambda e: e['id'] in self.db_ids,
    #             self.sabdab_entries
    #         )
    #     )

        # print("printed from _load_structures self.sabdab_entries", self.sabdab_entries)
# [{'id': '6qgy_B__A', 'pdbcode': '6qgy', 'H_chain': 'B', 'L_chain': None, 'ag_chains': ['A'], 'ag_type': 'protein', 'ag_name': 'outer membrane protein assembly factor bama', 'date': datetime.datetime(2019, 1, 14, 0, 0), 'resolution': 2.509, 'method': 'X-RAY DIFFRACTION', 'scfv': False}, {'id': '6qgy_B__A', 'pdbcode': '6qgy', 'H_chain': 'B', 'L_chain': None, 'ag_chains': ['A'], 'ag_type': 'protein', 'ag_name': 'outer membrane protein assembly factor bama', 'date': datetime.datetime(2019, 1, 14, 0, 0), 'resolution': 2.509, 'method': 'X-RAY DIFFRACTION', 'scfv': False}, {'id': '4c58_B__A', 'pdbcode': '4c58', 'H_chain': 'B', 'L_chain': None, 'ag_chains': ['A'], 'ag_type': 'protein', 'ag_name': 'cyclin-g-associated kinase', 'date': datetime.datetime(2013, 9, 10, 0, 0), 'resolution': 2.16, 'method': 'X-RAY DIFFRACTION', 'scfv': False}, {'id': '4c58_B__A', 'pdbcode': '4c58', 'H_chain': 'B', 'L_chain': None, 'ag_chains': ['A'], 'ag_type': 'protein', 'ag_name': 'cyclin-g-associated kinase', 'date': datetime.datetime(2013, 9, 10, 0, 0), 'resolution': 2.16, 'method': 'X-RAY DIFFRACTION', 'scfv': False}, {'id': '7rmh_N__AB', 'pdbcode': '7rmh', 'H_chain': 'N', 'L_chain': None, 'ag_chains': ['A', 'B'], 'ag_type': 'protein | protein', 'ag_name': 'guanine nucleotide-binding protein g(s) subunit alphaisoforms short  | guanine nucleotide-binding protein g(i)/g(s)/g(t) subunitbeta-1 ', 'date': datetime.datetime(2021, 7, 27, 0, 0), 'resolution': 3.1, 'method': 'ELECTRON MICROSCOPY', 'scfv': False}, {'id': '7rmh_N__AB', 'pdbcode': '7rmh', 'H_chain': 'N', 'L_chain': None, 'ag_chains': ['A', 'B'], 'ag_type': 'protein | protein', 'ag_name': 'guanine nucleotide-binding protein g(s) subunit alphaisoforms short  | guanine nucleotide-binding protein g(i)/g(s)/g(t) subunitbeta-1 ', 'date': datetime.datetime(2021, 7, 27, 0, 0), 'resolution': 3.1, 'method': 'ELECTRON MICROSCOPY', 'scfv': False}, {'id': '6qgy_D__C', 'pdbcode': '6qgy', 'H_chain': 'D', 'L_chain': None, 'ag_chains': ['C'], 'ag_type': 'protein', 'ag_name': 'outer membrane protein assembly factor bama', 'date': datetime.datetime(2019, 1, 14, 0, 0), 'resolution': 2.509, 'method': 'X-RAY DIFFRACTION', 'scfv': False}, {'id': '6qgy_D__C', 'pdbcode': '6qgy', 'H_chain': 'D', 'L_chain': None, 'ag_chains': ['C'], 'ag_type': 'protein', 'ag_name': 'outer membrane protein assembly factor bama', 'date': datetime.datetime(2019, 1, 14, 0, 0), 'resolution': 2.509, 'method': 'X-RAY DIFFRACTION', 'scfv': False}]

    # @property
    # def _structure_cache_path(self):
    #     return os.path.join(self.processed_dir, 'structures.lmdb')
    @property
    def _structure_cache_path(self):
        filename = f"structures_{self.split}_fold_{self.fold}.lmdb"  # This will create a filename like structures_train.lmdb, structures_val.lmdb, etc.
        return os.path.join(self.processed_dir, filename)

        
    def _preprocess_structures(self):
        lmdb_exists = os.path.exists(self._structure_cache_path) and os.path.exists(self._structure_cache_path + '-ids')

        # Skip preprocessing if LMDB database already exists
        if lmdb_exists:
            print("LMDB database already exists, skipping preprocessing.")
            return
        tasks = []
        failed_entries = []  # List to store the entries that failed to preprocess,ADDED

        for entry in self.sabdab_entries:

            if not self.special_filter:
                if entry['pdbcode'] not in self.entries_to_process:
                    continue  # Skip entries not in the current split
                pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode'])) # set it to all_structures/chothia
            if self.special_filter:
                if entry['entry_id'] not in self.entries_to_process:
                    continue  # Skip entries not in the current split
                old_pdb_path = entry['pdb_data_path']
                file_name = os.path.basename(old_pdb_path)
                pdb_path = os.path.join(self.chothia_dir,file_name)
                # print(pdb_path)
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            # 'id': entry['id'],
            tasks.append({
                'id': entry['entry_id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        data_list = []
        for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'):
            try:
                data = preprocess_sabdab_structure(task)
                if data is not None:
                    data_list.append(data)
            except Exception as e:
                logging.error(f"Error processing entry: {task['id']}, {e}")
                failed_entries.append(task['id'])
        # print("data_list", len(data_list))
        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)


    ############
        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )
        
        # print("datalist",len(data_list))
        # print("ids",ids[:10]) #ids ['6qgy_B__A', '6qgy_B__A', '4c58_B__A', '4c58_B__A', '7rmh_N__AB', '7rmh_N__AB', '6qgy_D__C', '6qgy_D__C']
    #     print("failed to process entries", failed_entries)
    #     return failed_entries


    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        try:
            subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)
        except:
            print("Error with mmseqs, pass")
            pass

    # def _preprocess_structures(self):
    #     tasks = []
    #     for entry in self.sabdab_entries:
    #         pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
    #         if not os.path.exists(pdb_path):
    #             logging.warning(f"PDB not found: {pdb_path}")
    #             continue
    #         tasks.append({
    #             'id': entry['id'],
    #             'entry': entry,
    #             'pdb_path': pdb_path,
    #         })  

    #     data_list = joblib.Parallel(
    #         n_jobs = max(joblib.cpu_count() // 2, 1),
    #     )(
    #         joblib.delayed(preprocess_sabdab_structure)(task)
    #         for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
    #     )
    #     print("print at _preprocess_structures")
    #     # print(data_list)

    #     db_conn = lmdb.open(
    #         self._structure_cache_path,
    #         map_size = self.MAP_SIZE,
    #         create=True,
    #         subdir=False,
    #         readonly=False,
    #     )
    #     #ADDED MELISSA
    #     # self.db_conn = db_conn
    #     # if db_conn:
    #     #     print("db_conn is not NONE")
    #     # else:
    #     #     print("db_conn is None, why?")
    #     ids = []
    #     with db_conn.begin(write=True, buffers=True) as txn:
    #         for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
    #             if data is None:
    #                 continue
    #             ids.append(data['id'])
    #             # print("ids",ids)
    #             # print("data['id']",data['id'])
    #             txn.put(data['id'].encode('utf-8'), pickle.dumps(data))
    #     print("ids", ids)

    #     with open(self._structure_cache_path + '-ids', 'wb') as f:
    #         pickle.dump(ids, f)


    def _load_split(self):
        assert self.split in ('train', 'val', 'test')

        if self.special_filter:
            print("entered the special filter if")
            if self.split == 'val':
                # ids_val = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.valid_entries]
                ids_val = self.valid_entries
                # self.ids_in_split = self.valid_entries[:3]
                self.ids_in_split = ids_val
                # print("self.ids_in_split", self.ids_in_split)
            elif self.split == 'train':
                # ids_train = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.train_entries]
                ids_train = self.train_entries
                self.ids_in_split = ids_train
        else: #original implementation
            ids_test = [entry['id'] for entry in self.sabdab_entries if entry['ag_name'] in TEST_ANTIGENS]
            test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])
            ids_train_val = [entry['id'] for entry in self.sabdab_entries if self.id_to_cluster[entry['id']] not in test_relevant_clusters]
            random.Random(self.split_seed).shuffle(ids_train_val)
            if self.split == 'test':
                self.ids_in_split = ids_test
            elif self.split == 'val':
                self.ids_in_split = ids_train_val[:20]
            else:
                self.ids_in_split = ids_train_val[20:]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    # def get_structure(self, id):
    #     # print(id)
    #     # print(self.db_ids)
    #     if id not in self.db_ids:
    #         raise ValueError(f"ID {id} not found in database IDs")


        
    #     self._connect_db()
    #     # print("self.db_conn", self.db_conn)
    #     with self.db_conn.begin() as txn:
    #         # encoded_pdbcode = pdbcode.encode()
    #         # print(encoded_pdbcode)
    #         encoded_id = id.encode()
    #         # print("encoded_id", encoded_id)
    #         data = txn.get(encoded_id)
    #         if data is None:
    #             raise ValueError(f"Data not found for ID: {id}")
    #         return pickle.loads(data)

    def get_structure(self, id):
        try:
            if id not in self.db_ids:
                raise ValueError(f"ID {id} not found in database IDs")
            
            self._connect_db()
            with self.db_conn.begin() as txn:
                encoded_id = id.encode()
                data = txn.get(encoded_id)
                if data is None:
                    raise ValueError(f"Data not found for ID: {id}")

                return pickle.loads(data)

        except ValueError as e:
            # Log the error and return None or dummy data
            print(f"Warning: Skipping ID {id} - {e}")
            return None  # You could also return some dummy data here if appropriate


    def __len__(self):
        return len(self.ids_in_split)

    # def __getitem__(self, index):
    #     id = self.ids_in_split[index]
    #     data = self.get_structure(id)
    #     if self.transform is not None:
    #         data = self.transform(data)
    #     return data

    def __getitem__(self, index):
        try:
            id = self.ids_in_split[index]
            data = self.get_structure(id)
            if self.transform is not None:
                data = self.transform(data)
        except Exception as e:
            logging.error(f"Error processing index {index} (ID: {id}): {e}")
            return None  # Or return some default value
        return data


# class SAbDabDataset2(Dataset):

#     MAP_SIZE = 32*(1024*1024*1024)  # 32GB
#     def __init__(
#         self, 
#         summary_path,  
#         chothia_dir,   
#         processed_dir,  
#         split,        
#         split_seed,   
#         transform,     
#         reset,
#         special_filter
#     ):
#         super().__init__()


#         print("conducting fold = ", self.fold)
#         self.special_filter = special_filter
#         self.summary_path = summary_path
#         self.chothia_dir = chothia_dir
#         self.split=split
#         self.split_seed=split_seed

#         if not os.path.exists(chothia_dir):
#             raise FileNotFoundError(
#                 f"SAbDab structures not found in {chothia_dir}. "
#                 "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
#             )
        

#         # Decide which entries to process based on the split
#         if self.split == 'train':
#             self.entries_to_process = self.train_entries
#         elif self.split == 'val':
#             self.entries_to_process = self.valid_entries
#         # elif split == 'test':
#         #     self.entries_to_process = self.test_entries
#         else:
#             raise ValueError("Invalid split specified")

#         self.processed_dir = processed_dir
#         os.makedirs(processed_dir, exist_ok=True)

#         self.reset = reset
    
#         self.sabdab_entries = None
#         self._load_sabdab_entries()

#         self.db_conn = None
#         self.db_ids = None
#         self._load_structures(reset)

#         # self.clusters = None
#         # self.id_to_cluster = None
#         # self._load_clusters(reset)  #comment it and its functions if performing the special_filter = True

#         self.ids_in_split = None
#         self._load_split()

#         self.transform = transform  # You can provide a transformation function
    

#         print("self.sabdab_entries",len(self.sabdab_entries),chothia_dir)
#         # print(len(self.sabdab_entries))

#     def _load_sabdab_entries(self):
#         print("performing_load_sabdab_entries")
#         df = pd.read_csv(self.summary_path, sep='\t')
#         # print("df", len(df))
#         entries_all = []
#         for i, row in tqdm(
#             df.iterrows(), 
#             dynamic_ncols=True, 
#             desc='Loading entries',
#             total=len(df),
#         ):
#             entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
#                 pdbcode = row['pdb'],
#                 H = nan_to_empty_string(row['Hchain']),
#                 L = nan_to_empty_string(row['Lchain']),
#                 Ag = ''.join(split_sabdab_delimited_str(
#                     nan_to_empty_string(row['antigen_chain'])
#                 ))
#             )
#             ag_chains = split_sabdab_delimited_str(
#                 nan_to_empty_string(row['antigen_chain'])
#             )
#             resolution = parse_sabdab_resolution(row['resolution'])
#             entry = {
#                 'id': entry_id,
#                 'pdbcode': row['pdb'],
#                 'H_chain': nan_to_none(row['Hchain']),
#                 'L_chain': nan_to_none(row['Lchain']),
#                 'ag_chains': ag_chains,
#                 'ag_type': nan_to_none(row['antigen_type']),
#                 'ag_name': nan_to_none(row['antigen_name']),
#                 'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
#                 'resolution': resolution,
#                 'method': row['method'],
#                 'scfv': row['scfv'],
#             }

#             if entry['pdbcode'] in self.all_:
#                 entries_all.append(entry)
#             # Filtering, ADDED
#             if not self.special_filter:
#                 if ((entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None) and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)):
#                     entries_all.append(entry)
#             elif entry['pdbcode'] in self.all_:
#                 entries_all.append(entry)

#         self.sabdab_entries = entries_all


#         processed_entries = []
#         for entry in self.sabdab_entries:
#             processed_entries.append(entry)


#     def _load_structures(self, reset):
#         # if os.path.exists(self._structure_cache_path) and not reset:
#         if not reset:
#             # LMDB database exists and reset is not requested, so load from LMDB
#             with open(self._structure_cache_path + '-ids', 'rb') as f:
#                 self.db_ids = pickle.load(f)

#             if self.db_ids is None:
#                 raise ValueError("Failed to load db_ids from file")

#             self.sabdab_entries = list(
#                 filter(
#                     lambda e: e['id'] in self.db_ids,
#                     self.sabdab_entries
#                 )
#             )
#         elif not os.path.exists(self._structure_cache_path):
#             # LMDB database does not exist, so preprocess structures
#             self._preprocess_structures()


#     @property
#     def _structure_cache_path(self):
#         filename = f"structures_{self.split}_fold_{self.fold}.lmdb"  # This will create a filename like structures_train.lmdb, structures_val.lmdb, etc.
#         return os.path.join(self.processed_dir, filename)

        
#     def _preprocess_structures(self):
#         lmdb_exists = os.path.exists(self._structure_cache_path) and os.path.exists(self._structure_cache_path + '-ids')

#         # Skip preprocessing if LMDB database already exists
#         if lmdb_exists:
#             print("LMDB database already exists, skipping preprocessing.")
#             return
#         tasks = []
#         failed_entries = []  # List to store the entries that failed to preprocess,ADDED

#         for entry in self.sabdab_entries:
#             if entry['pdbcode'] not in self.entries_to_process:
#                 continue  # Skip entries not in the current split

#             pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
#             if not os.path.exists(pdb_path):
#                 logging.warning(f"PDB not found: {pdb_path}")
#                 continue
#             tasks.append({
#                 'id': entry['id'],
#                 'entry': entry,
#                 'pdb_path': pdb_path,
#             })

#         data_list = []
#         for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'):
#             try:
#                 data = preprocess_sabdab_structure(task)
#                 if data is not None:
#                     data_list.append(data)
#             except Exception as e:
#                 logging.error(f"Error processing entry: {task['id']}")
#                 failed_entries.append(task['id'])
#         # print("data_list", len(data_list))
#         db_conn = lmdb.open(
#             self._structure_cache_path,
#             map_size = self.MAP_SIZE,
#             create=True,
#             subdir=False,
#             readonly=False,
#         )
#         ids = []
#         with db_conn.begin(write=True, buffers=True) as txn:
#             for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
#                 ids.append(data['id'])
#                 txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

#         with open(self._structure_cache_path + '-ids', 'wb') as f:
#             pickle.dump(ids, f)


#     ############
#         with open(self._structure_cache_path + '-ids', 'rb') as f:
#             self.db_ids = pickle.load(f)
#         self.sabdab_entries = list(
#             filter(
#                 lambda e: e['id'] in self.db_ids,
#                 self.sabdab_entries
#             )
#         )
        

#     @property
#     def _cluster_path(self):
#         return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

#     def _load_clusters(self, reset):
        
#         if not os.path.exists(self._cluster_path) or reset:
#             self._create_clusters()

#         clusters, id_to_cluster = {}, {}
#         with open(self._cluster_path, 'r') as f:
#             for line in f.readlines():
#                 cluster_name, data_id = line.split()
#                 if cluster_name not in clusters:
#                     clusters[cluster_name] = []
#                 clusters[cluster_name].append(data_id)
#                 id_to_cluster[data_id] = cluster_name
#         self.clusters = clusters
#         self.id_to_cluster = id_to_cluster

#     def _create_clusters(self):
#         cdr_records = []
#         for id in self.db_ids:
#             structure = self.get_structure(id)
#             if structure['heavy'] is not None:
#                 cdr_records.append(SeqRecord.SeqRecord(
#                     Seq.Seq(structure['heavy']['H3_seq']),
#                     id = structure['id'],
#                     name = '',
#                     description = '',
#                 ))
#             elif structure['light'] is not None:
#                 cdr_records.append(SeqRecord.SeqRecord(
#                     Seq.Seq(structure['light']['L3_seq']),
#                     id = structure['id'],
#                     name = '',
#                     description = '',
#                 ))
#         fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
#         SeqIO.write(cdr_records, fasta_path, 'fasta')

#         cmd = ' '.join([
#             'mmseqs', 'easy-cluster',
#             os.path.realpath(fasta_path),
#             'cluster_result', 'cluster_tmp',
#             '--min-seq-id', '0.5',
#             '-c', '0.8',
#             '--cov-mode', '1',
#         ])
#         try:
#             subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)
#         except:
#             print("Error with mmseqs, pass")
#             pass


#     def _load_split(self):
#         assert self.split in ('train', 'val', 'test')

#         if self.special_filter:
#             print("entered the special filter if")
#             if self.split == 'val':
#                 ids_val = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.valid_entries]
#                 # self.ids_in_split = self.valid_entries[:3]
#                 self.ids_in_split = ids_val
#                 # print("self.ids_in_split", self.ids_in_split)
#             elif self.split == 'train':
#                 ids_train = [entry['id'] for entry in self.sabdab_entries if entry['pdbcode'] in self.train_entries]
#                 self.ids_in_split = ids_train
#         else: #original implementation
#             ids_test = [entry['id'] for entry in self.sabdab_entries if entry['ag_name'] in TEST_ANTIGENS]
#             test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])
#             ids_train_val = [entry['id'] for entry in self.sabdab_entries if self.id_to_cluster[entry['id']] not in test_relevant_clusters]
#             random.Random(self.split_seed).shuffle(ids_train_val)
#             if self.split == 'test':
#                 self.ids_in_split = ids_test
#             elif self.split == 'val':
#                 self.ids_in_split = ids_train_val[:20]
#             else:
#                 self.ids_in_split = ids_train_val[20:]

#     def _connect_db(self):
#         if self.db_conn is not None:
#             return
#         self.db_conn = lmdb.open(
#             self._structure_cache_path,
#             map_size=self.MAP_SIZE,
#             create=False,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )

#     def get_structure(self, id):
#         if id not in self.db_ids:
#             raise ValueError(f"ID {id} not found in database IDs")
        
#         self._connect_db()
#         # print("self.db_conn", self.db_conn)
#         with self.db_conn.begin() as txn:
#             # encoded_pdbcode = pdbcode.encode()
#             # print(encoded_pdbcode)
#             encoded_id = id.encode()
#             # print("encoded_id", encoded_id)
#             data = txn.get(encoded_id)
#             if data is None:
#                 raise ValueError(f"Data not found for ID: {id}")
#             return pickle.loads(data)

#     def __len__(self):
#         return len(self.ids_in_split)

#     def __getitem__(self, index):
#         id = self.ids_in_split[index]
#         data = self.get_structure(id)
#         if self.transform is not None:
#             data = self.transform(data)
#         return data



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='/ibex/user/rioszemm/diffab/data/processed_Ab_Nb_december') #Modify
    parser.add_argument('--reset', action='store_true', default=True)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(
        processed_dir=args.processed_dir,
        split=args.split, 
        reset=args.reset
    )
    # print(dataset[0])
    print(f"{len(dataset)}, num of clusters {len(dataset.clusters)}")
    

    #NAnobodies
    # self.valid_entries = ['6eqi', '7vq0', '7qiv', '7paf', '8bpk', '7klw', '8dfl', '7mfu', '7ust', '8cyb', '7l6v', '7zw1', '7kdu', '6uc6', '7nk4', '8emy', '7ph4', '7kbk', '6obm', '7yag', '4lhq', '7sla', '6ze1', '6z20', '8gni', '7can', '5h8o', '5ivn', '1zv5', '5boz', '8emz', '4y7m', '7s7r', '5m2j', '7tpr', '4x7d', '4pgj', '3cfi', '4fhb', '5vl2', '4bfb', '5vnw', '6app', '4pou', '3jbc', '7nx0', '6qpg', '7n9t', '5vm6', '6gjq', '7z85', '5vxk', '5ja8', '7vfa', '6zxn', '4i13', '8cxn', '5fhx', '7kji', '4wen', '4hjj', '8en3', '3rjq', '7z7x', '3qsk', '2p4a', '7n0r', '6obo', '7nki', '7y9t', '5nbl', '4cdg', '1op9', '6qv1', '6oq7', '6lz2', '5f7l', '8sbb', '6xw5', '7ar0', '7sl9', '6qv2', '6i6j', '6quz', '1kxv', '7pc0', '6oca', '5lhp', '6tyl', '7kkk', '6x05', '8cxr', '7x7e', '7zk1', '7vke', '6o8d', '7x2l', '7o31', '4w6y', '3k7u', '6qx4', '5f21', '7r4i', '7qbg', '7dst', '7saj', '6v80', '5j1t', '7xrp', '5ip4', '7epb', '5m30', '4m3k']
    # self.test_entries = ['6qup', '7qbe', '7f5g', '7nkt', '7b14', '7olz', '7zfb', '7wd2', '6fyw', '8dlx', '5nbd', '6b20', '7b17', '7lzp', '6u55', '5tsj', '7p2d', '8en2', '7ldj', '6qgy', '1g6v', '6xw7', '6oq5', '3jbd', '5m14', '6fv0', '6i2g', '5m2i', '6obe', '5eul', '6war', '4nc1', '6ruv', '2xv6', '6nfj', '7kkl', '5g5x', '6rvc', '7sqp', '5fuc', '5sv3', '7aej', '1rjc', '7apj', '7aqz', '8en6', '6c9w', '6h6y', '6gs1', '7nvm', '5y80', '6q6z', '7tgf', '7tgi', '2x89', '5mwn', '7qne']
    # self.train_entries = ['7k84', '8c8p', '5hvh', '7zg0', '4w6x', '6ibb', '7n9v', '5o0w', '7b2p', '8h3y', '2p42', '1kxt', '4krm', '7q3q', '6fyt', '5nqw', '7p5w', '6fyu', '7unz', '8cyj', '7r1z', '8ont', '6jb2', '7aqy', '7nbb', '6b73', '7aqg', '7tjc', '7zkw', '7pqg', '7phq', '7a0v', '6c5w', '6cwg', '5m15', '7th2', '3k74', '7w1s', '6u51', '1ri8', '4pir', '7kjh', '2vyr', '4hem', '8gz5', '7r74', '7pa5', '6h71', '7m1h', '5l21', '6o3c', '5jds', '6v7z', '8cy9', '7s2r', '6x07', '6sc7', '4kro', '6f5g', '7q6z', '7d5q', '7p5y', '6i8h', '7fh0', '7rmi', '5vm0', '4hep', '8dtn', '7fat', '8cy6', '7a4t', '7ri1', '7eow', '7p7a', '5vxj', '3p0g', '4krp', '7a4y', '4gft', '6qgx', '5vxm', '7yzi', '4p2c', '7nfq', '7cz0', '1qd0', '5mzv', '6x06', '7qbf', '7r24', '6zg3', '8en4', '4grw', '7ngh', '7b27', '6ul4', '7qvk', '7p16', '7b5g', '4lgs', '7a48', '7nik', '6u12', '6u54', '4z9k', '6rpj', '3k3q', '4i1n', '6zpl', '7sp6', '6oq6', '8cy7', '5tok', '6uht', '7nxx', '7kgk', '7qjj', '4xt1', '7c8v', '4x7f', '4ksd', '6zrv', '1kxq', '5f1k', '8e99', '7z1b', '7n9e', '3k80', '7ozt', '8bev', '8b7w', '6gwp', '7nj5', '6sge', '5mp2', '4qo1', '5e5m', '5vaq', '5toj', '5o05', '7mjh', '6xzu', '7q3r', '8cwu', '3ezj', '7kbi', '7yit', '6fe4', '7r4r', '7wd1', '7fau', '4kfz', '5tjw', '5jqh', '4lgp', '7d2z', '5dfz', '4w2q', '5mje', '7uny', '6gci', '6os1', '8bb7', '4wem', '7k65', '6uft', '4nc2', '7qe5', '5usf', '7zra', '6zwk', '4laj', '7s2s', '4n1h', '7e53', '7jvb', '7jwb', '7d30', '6h6z', '5e7f', '7vfb', '7th3', '8g2y', '6vi4', '6gju', '4w2o', '8f8w', '7n0i', '6h15', '5vxl', '5ihl', '8dqu', '7zmv', '7n9c', '8e0e', '5lwf', '4i0c', '7wn1', '1mvf', '8b41', '7p78', '6rty', '5imm', '6x08', '4orz', '6h02', '6tej', '7upm', '5f1o', '5td8', '6h16', '6obg', '7lvw', '5jmo', '5o03', '6qgw', '7mdw', '7nft', '3v0a', '7me7', '8ew6', '7rga', '6lr7', '2bse', '2x6m', '7nqa', '8en1', '4dk6', '6hhu', '3k81', '6ocd', '7fg3', '7xk9', '8cyc', '7p60', '5ja9', '7o3b', '7wn0', '3stb', '6z1v', '5n88', '5c2u', '7wuj', '6ul6', '7a50', '7aqx', '7kc9', '7wpf', '7z1e', '7sai', '7q6c', '7omm', '6vbg', '5bop', '5f93', '7dv4', '7sp9', '7e6u', '6xw6', '4ios', '1zvy', '6dbf', '5hgg', '7a4d', '8h3x', '6i8g', '5my6', '7usv', '7oap', '6gkd', '5hvf', '6ssp', '7te8', '8cyd', '7om4', '5fv2', '8cxq', '7c8w', '4lhj', '8dtu', '7wki', '6knm', '6qtl', '7my3', '7qbd', '7uia', '7oau', '4yga', '5ovw', '6ey0', '6obc', '7nil', '7t5f', '7o06', '7p6k', '7rby', '1zvh', '4eig', '7d4b', '7kd0', '7na9', '7whj', '5omn', '8dyp', '7czd', '6xw4', '7rnn', '1mel', '5e0q', '6r7t', '8di5', '7km5', '8en5', '8bzy', '7php', '5lhr', '3p9w', '5u4m', '7b2q', '7b2m', '6f2g', '6ssi', '6itq', '8en0', '6ir1', '7a29', '8b01', '7xtp', '6yu8', '7kd2', '7jhg', '7x2j', '5hm1', '7ns6', '6yz7', '3k1k', '4s10', '7z1c', '5e1h', '4weu', '4nc0', '7ubx', '5o02', '7dss', '7pqq', '7now', '7azb', '7ocj', '5m2m', '7x4i', '6hhd', '7r4q', '7bc6', '4w6w', '7nfr', '6ey6', '7nj7', '7ocy', '7f5h', '6n50', '7xqv', '4ocn', '7nmu', '8c3l', '8bf4', '6v7y', '7voa', '6j7w', '6rqm', '6z6v', '7r73', '6s0y', '4eiz', '6x04', '5da0', '7k7y', '7nj3', '7x2m', '7nqk', '5gxb', '7uby', '6ysq', '4c59', '6waq', '6hjy', '5c3l', '4lgr', '7my2', '7z1a', '7qcq', '6zcz', '3g9a', '4mqs', '8a67', '6csy']



    #     # Antibodies No Haptens, carbohydrates etc
    #     self.test_entries = ['3ffd', '4etq', '4ki5', '2cmr', '4qci', '1iqd', '1uj3', '5en2', '5nuz', '2ypv', '4cmh', '2b2x', '1osp', '3k2u', '2xwt', '4fqj', '3rkd', '2adf', '4ot1', '5l6y', '3bn9', '3hi6', '3l95', '3s35', '2xqy', '3o2d', '5f9o', '5j13', '5ggs', '4ffv', '4ydk', '1fe8', '4lvn', '4dtg', '5bv7', '1w72', '5d96', '4jlr', '7dm2', '8fbw', '6vel', '7qf1', '6nz7', '6oe4', '4wv1', '6bfs', '5k9q', '4jo1', '7sbd', '3p30', '8cyh', '1i9r', '6xkp', '5tfw', '8fa7', '6kyz', '1frg', '7jkt', '6oor', '5t1d', '5jq6', '7mzh', '7chp', '6t3f', '3iu3', '6mek', '5xku', '3bsz', '5xj3', '2vis', '7pr0', '6n6b', '6b5r', '4p3c', '4d3c', '7kez', '5fb8']
    #     self.train_entries = ['6r8x', '2bdn', '5gjt', '5usl', '7fjc', '1kcs', '5e94', '6oz2', '5csz', '3h0t', '8f6o', '6q20', '5u3n', '5t3z', '3pgf', '6mqr', '7l7r', '5eii', '6n5b', '3ztj', '6obd', '1f58', '2zuq', '6u6u', '6x8q', '3hr5', '7o33', '4xgz', '8df1', '2v17', '6cw3', '5bo1', '1cz8', '6bli', '1nmc', '5w06', '7cu4', '2jix', '6oel', '2y6s', '3s36', '5dub', '7jks', '1fbi', '5bjz', '7wvl', '4hcr', '7n4l', '6h06', '6wty', '5vl3', '6z7y', '7kq7', '5mub', '7n3c', '6xm2', '7rt9', '4n8c', '6ldy', '7rxp', '7kmh', '4oqt', '8f60', '5vpl', '6nyq', '3fmg', '7r6x', '1h0d', '8dao', '5njd', '4jzj', '5tzu', '6ayz', '8fax', '7dpm', '8smt', '8ahn', '5v6m', '7t25', '7uvf', '7xsb', '5f3h', '5tkk', '3sge', '4g80', '7orb', '4kvn', '5jhl', '6p60', '5th9', '4xnz', '3vrl', '6bfq', '4rgn', '5myo', '7fcp', '3b2u', '7wg3', '5wna', '5kvg', '7ul0', '6wf0', '5u3j', '6xli', '2h1p', '6qig', '7e72', '6wg0', '7t86', '5vic', '5tkj', '4f15', '5a3i', '6t9e', '6vry', '7chf', '6lgw', '5x0t', '6aru', '7kpb', '8dcy', '4dke', '5sx4', '8gb5', '7cho', '5uoe', '5umn', '6gk8', '7q4q', '7ufo', '5xhv', '6mtq', '3ncy', '2hh0', '3sqo', '7tuf', '7o9s', '6o2b', '4aei', '4z5r', '5wko', '8ath', '6nnj', '6cbv', '7ue9', '7sg6', '7pi2', '7rah', '3nps', '1qle', '6m38', '3vi3', '7jtg', '8dn6', '6pxr', '8d36', '7tp3', '6ii4', '7lf8', '7u8c', '7s0x', '7ssc', '6x78', '8cwj', '6dfj', '5t33', '5vk2', '5fgc', '4uu9', '7ce2', '5w08', '7beh', '8dgu', '4gxu', '8eoo', '4jpv', '5utz', '7txw', '7yds', '5w3p', '7cgw', '3pnw', '8db4', '4am0', '6b0e', '7unb', '7s4s', '6x7w', '7dk0', '7ps7', '6oej', '8d6z', '7u8e', '2eh8', '7e5y', '3so3', '6n7j', '3i50', '4glr', '7zxk', '2qqk', '4g6a', '4u6h', '5f9w', '8f6l', '7k75', '5czx', '6umx', '6xpy', '7y3j', '4o58', '5ywy', '7s0b', '7x8t', '5nh3', '5ukr', '7r8u', '4xvt', '7mzl', '7pry', '7z2m', '4rx4', '7cjf', '7wbz', '7kql', '5cws', '6o9i', '6p8n', '3iet', '7x8q', '5tlj', '5tpn', '7shy', '6bzy', '6tyb', '5te6', '4q6i', '7l2c', '4jb9', '4n9g', '7mfr', '7phu', '4v1d', '6iap', '7cdj', '7kpg', '7xjf', '3sdy', '6id4', '7ox1', '8hit', '4m8q', '7djz', '7kmg', '6dkj', '8gng', '5b3j', '3nh7', '7tp4', '4xvu', '6wfx', '5dmg', '4tsa', '6flb', '7lr3', '5occ', '5lqb', '6vbo', '3ab0', '6oy4', '1sy6', '4hg4', '8d9y', '4zs6', '4s1r', '5kaq', '2fjh', '5tbd', '7e5o', '6vc9', '6k0y', '6mqm', '6o9h', '6o41', '7zfd', '7s5q', '7kmi', '1jps', '6mqe', '6xp6', '4zpv', '4j8r', '4jzn', '7mmo', '1oak', '2jel', '5dfv', '8dxt', '7or9', '8ib1', '6zvf', '8dn7', '5w5z', '5wux', '5x8l', '4q0x', '4rav', '6w16', '1u8p', '7rew', '6gv1', '7ps4', '1zea', '1kc5', '7beo', '4hpy', '3tt3', '6b0h', '5y2l', '1ktr', '6bkd', '5d1q', '8ol9', '7e9b', '4ydi', '8fan', '1egj', '8euq', '6ii9', '3c09', '8gq1', '8as0', '4ydv', '4s1s', '7nks', '7bbj', '7jmo', '6wo3', '6w4m', '1ob1', '1eo8', '6t3j', '7kfv', '7o30', '7lcv', '6wg1', '1orq', '7ahv', '1a3r', '5myx', '8fdo', '6uyg', '7mzf', '4rwy', '6bdz', '5l0q', '6fy3', '8dtr', '5kan', '3r1g', '3mlz', '7sue', '7prz', '6plh', '6aq7', '7rd5', '1xgy', '8djg', '7s8h', '6e56', '4l5f', '7raq', '5udc', '6ct7', '2xra', '8gpy', '4yqx', '3n85', '6n81', '2vxq', '6pdu', '5uem', '4hf5', '8hpk', '5d1x', '6wqo', '8gho', '4zso', '6cxy', '6wvz', '6pdx', '6ule', '7vzt', '7zfb', '8bbo', '6j6y', '5fgb', '4o9h', '4tul', '7re7', '2vwe', '6xlz', '6h3u', '8fa9', '6pcu', '6yxd', '7uvs', '4rfo', '6d11', '8hpj', '5eu7', '6bzw', '7lr4', '6rps', '3be1', '7lfb', '4n0y', '6iea', '6hhc', '5dur', '6o3b', '7si2', '2j5l', '5w23', '7ps2', '7rfb', '6ieb', '6iek', '6hxw', '7ucf', '6x1w', '5i9q', '7st5', '6x1v', '5w6g', '4yfl', '7e86', '7ean', '6a67', '6xq0', '5ikc', '6b0g', '8gpu', '7yv1', '1qfu', '6vgr', '4kuc', '7x63', '5ush', '5d8j', '6sf6', '1ndm', '6df2', '7t7b', '6ewb', '3thm', '3uc0', '4f2m', '7xeg', '6b3m', '7dk2', '2h32', '5ocx', '6ous', '7tri', '8ag1', '6x1u', '2nr6', '4yhz', '6db6', '6xzw', '4qhu', '6mns', '7vyt', '4edx', '7bxv', '3tt1', '7s13', '4xi5', '4om0', '4uao', '6mqc', '4ht1', '5hdq', '7kn4', '7fah', '7shu', '7yck', '6j11', '4rgm', '1fj1', '1e4x', '5w1k', '7mzk', '1uwx', '6pzf', '1vfb', '6wfy', '8elo', '2q8a', '3kr3', '6xxv', '7ps5', '3ks0', '7y8j', '1yqv', '5vqm', '8fdc', '4nnp', '7x9e', '1bj1', '6q18', '6u36', '2aep', '7vyr', '7ul1', '6fy1', '7z0y', '5lbs', '6woz', '5dum', '8eee', '2qr0', '7qu2', '1ai1', '7sfx', '8gb7', '6n5d', '6dc5', '6nha', '6vmj', '4dn4', '5if0', '6ieq', '4wuu', '7zf3', '7jmp', '5t6p', '8eb2', '8byu', '4u1g', '6cyf', '6yxk', '5hj3', '7jix', '5drz', '4ffy', '6eay', '4y5v', '1kcr', '3ubx', '6mlk', '7u0e', '8aon', '7chz', '6ion', '1ynt', '6dc3', '3qnz', '8gf2', '4h88', '6dca', '5x08', '7e7e', '7kf1', '1ors', '5o1r', '7dha', '6mvl', '1yjd', '6o39', '6xpx', '1v7n', '8dk6', '4hfu', '7u0a', '5veb', '7df1', '5tz2', '5epm', '5tlk', '2r0l', '6p3s', '3liz', '8fba', '6txz', '8bbn', '7tug', '5czv', '4jhw', '6cmg', '7coe', '7te1', '1xgq', '6p9h', '6vy4', '8dnn', '6pec', '6f7t', '7pa9', '7yru', '1z3g', '6cse', '8av9', '1ggi', '5jz7', '6vjt', '7v64', '7bh8', '5y11', '8j7e', '5tih', '6lxj', '7seg', '6pds', '6p8m', '5u7o', '4mwf', '7c94', '7cj2', '6d2p', '2g5b', '4bkl', '5cjx', '4k2u', '5h35', '4iof', '6n5e', '7uzd', '7qnw', '5n0a', '4jdt', '4zff', '7uxl', '6vi1', '6mi2', '5lsp', '5uqy', '4cad', '6h3t', '5esv', '7wnb', '5vcn', '7pi7', '8gh4', '4d9r', '6svl', '8f0i', '6glx', '7rd3', '1ahw', '6yxm', '7ps0', '6vbq', '6u9s', '6wit', '6vep', '6x4t', '6o3a', '5dsc', '1hez', '7lsf', '6bit', '7lja', '3g04', '5eoq', '3b9k', '7sts', '6kva', '3qg6', '8dtt', '7dm1', '6lyn', '4xxd', '2gsi', '4m5z', '1ztx', '5c0n', '7rdw', '5dlm', '5ggv', '6dc9', '3dvn', '7n0x', '7ufn', '7sqw', '6o3l', '8dfi', '3ulu', '5ob5', '7m7w', '5mhr', '6cez', '4dag', '5grj', '6att', '6jbt', '6i3z', '6xsw', '7bnv', '7xsc', '4n90', '6ba5', '8hn7', '4ubd', '6fgb', '2w9e', '6vvu', '2ny3', '3l5y', '7z0x', '7m3i', '6uoe', '7w71', '7jie', '5nj6', '6cdp', '6awo', '7eam', '5wb9', '7uvh', '4yk4', '7t87', '6wmw', '7jtf', '6w05', '8gy5', '8caf', '1mvu', '7pi3', '7x8p', '5k9o', '7amr', '5w4l', '1ezv', '6oz9', '6vca', '5tzt', '6kz0', '6o24', '8dwa', '6wir', '7ycn', '7yzj', '6pi7', '7bel', '7bbg', '5dup', '7ceb', '6db5', '6k65', '3kj6', '7lki', '8slb', '7fjs', '6vbp', '2j88', '7uvq', '1nfd', '4rrp', '7rfc', '7sjs', '6snc', '1nak', '1ndg', '5ifj', '5o4o', '6xq2', '8b9v', '4jre', '8h8q', '6vug', '5zia', '7y0c', '2r29', '8gb6', '7chs', '7ytn', '7uzc', '7x7o', '6wj1', '8c3v', '4yr6', '6wm9', '5mhs', '6o25', '4hj0', '6w51', '4bz2', '3hae', '7qu1', '5cbe', '8f0l', '4k3j', '7ukt', '6lz4', '6pbv', '8bse', '7xrz', '6p7h', '4jan', '5v6l', '7ox3', '6bf4', '7kyo', '5wtt', '6cwt', '3eff', '3v4u', '5ijk', '1ikf', '8ekf', '6w7s', '6hf1', '6yio', '5c6t', '6pis', '7pa8', '3se8', '6a4k', '7njz', '3hmx', '7zf8', '8dxu', '4jpw', '1adq', '4y5x', '6m58', '4edw', '7keo', '6h5n', '8e1g', '4cni', '6mnq', '7tcq', '7cr5', '7wsl', '8f5i', '6wzk', '6b0n', '6lz9', '2oz4', '6bzu', '7lfa', '6pdr', '6icc', '4leo', '7xik', '7quh', '7joo', '6xr0', '6ddv', '5ggr', '5ea0', '6i9i', '7uvi', '1e6j', '5cus', '8dcm', '7s4g', '8oxv', '8ee1', '1fsk', '6ktr', '7l7e', '5ldn', '7mdp', '7ycl', '6i04', '7np1', '5o4g', '4qww', '3wsq', '7pgb', '5vob', '7s7i', '5eoc', '7rxi', '6xqw', '4ypg', '5u8q', '4ydl', '4o4y', '5kvd', '5gzo', '1lk3', '8ivx', '7bq5', '6bzv', '5o14', '6nmt', '6phf', '6urh', '4wht', '7msq', '4oii', '2xqb', '4u6v', '3qum', '8ddk', '7klg', '1jrh', '4zfo', '7mzm', '6mqs', '1fpt', '3lqa', '7k66', '6jep', '4dkf', '6wno', '8d0a', '6a77', '6ii8', '3u9u', '8dfg', '5d1z', '7cm4', '2brr', '5vxr', '5y9c', '5kvf', '5fec', '7sem', '4hzl', '6was', '3gi8', '6iuv', '3h42', '7n0u', '3wkm', '7sl5', '6ddr', '6cw2', '8gpx', '2r0z', '4lkx', '6e4y', '6wtv', '8ee8', '6wo5', '5c8j', '6otc', '4hs6', '5vl7', '6x1t', '6elu', '4mxw', '3g6j', '2r0k', '4r3s', '5nmv', '7rm0', '7l5b', '8dw9', '7n4j', '5d72', '7ahu', '7y0v', '7rks', '7ejz', '3ujj', '6z06', '7e7y', '5mvz', '8ee5', '8e1p', '7wn2', '6iw0', '3idy', '1hh6', '6n8d', '3w2d', '7dc8', '6uig', '7rku', '4od2', '6q23', '6ln2', '7u2e', '4i3r', '8f5n', '3skj', '6azz', '8ivw', '6xkr', '8hn6', '2qhr', '6n16', '7mzg', '8fdd', '7yxu', '6ohg', '5c0r', '5gmq', '5wdf', '4i77', '5tud', '6uta', '6bb4', '4xmk', '5ye3', '6ppg', '4okv', '6vy5', '1qkz', '4ut6', '7kqh', '6erx', '4tnw', '3fn0', '5xwd', '6mtt', '7u5b', '7vux', '6adc', '7so5', '8f7z', '3u9p', '2hfg', '3v6o', '5zs0', '7uvo', '6bck', '5b71', '6ute', '7m51', '3efd', '7s07', '3qa3', '6jjp', '8ffe', '6wg2', '2a6k', '1qfw', '8be1', '6q1z', '8f9v', '6e3h', '4qti', '7n08', '5vlp', '5xbm', '6e4x', '7amq', '6phd', '6nmu', '6m3c', '7s11', '3bgf', '4p59', '6hx4', '6r2s', '5yoy', '6axl', '6o1f', '8dgv', '3lzf', '2zjs', '7o2z', '3nfp', '4r0l', '1tzh', '4rau', '2qsc', '3d85', '7vtj', '6p95', '4m1g', '7q0i', '6nmv', '7zmk', '6wx2', '7kyl', '1kb5', '5eor', '6xq4', '6uyn', '8fg0', '8fa6', '5mo9', '6plk', '8dgw', '5dwu', '3go1', '1ifh', '3eob', '3bae', '7kn3', '4ydj', '8eed', '6co3', '5vkd', '8f95', '3cvh', '4np4', '6rlo', '6o26', '4yue', '2uud', '7sbu', '5ibl', '4onf', '7rxj', '2j6e', '5y9f', '5x2o', '7rqq', '4lu5', '4plk', '7yk4', '4y5y', '7t0l', '8fas', '7mf1', '5bk2', '4dqo', '3jwo', '6mtp', '6d9w', '5t5n', '4j4p', '7bep', '7um3', '6h2y', '3e8u', '6wjl', '6ye3', '6xy2', '5usi', '7ly3', '7xy8', '7e7x', '2xtj', '6flc', '1za3', '7mzn', '6ldv', '7sx7', '3bky', '7wn8', '3zkn', '5ug0', '4ybq', '7n0a', '7bsc', '2nyy', '7t82', '7lm8', '5kn5', '6pze', '7st8', '6z2l', '4qnp', '1ken', '8f9u', '3o0r', '4ye4', '6hig', '3p0y', '6glw', '8dtx', '7ew5', '7n4i', '2i9l', '4zs7', '6bp2', '7qny', '4tsb', '3uji', '6cdm', '3lex', '4jg0', '6o23', '5wi9', '6ddm', '7rxl', '4bz1', '6nms', '6iec', '8el2', '8duz', '7r6w', '2arj', '5anm', '1nca', '6m3b', '5bk1', '4ut9', '5xcu', '6mg7', '5gzn', '1g9m', '7k8m', '1pkq', '3g5y', '7doh', '4xh2', '6pe9', '6yax', '8fat', '8dv6', '7rlw', '3t2n', '4yhp', '6c6z', '3u30', '1ejo', '6k7o', '3v7a', '8a44', '7s5p', '7lk4', '5vig', '4xvj', '7kqg']
    #     self.valid_entries = ['6urm', '2b1h', '6wn1', '6vjn', '6p8d', '3ifp', '6b0a', '7b0b', '7str', '6d0u', '7q6c', '4m1d', '7mnl', '5w0k', '4xak', '7s3m', '7d85', '6mhr', '7s1b', '8cz5', '7eyc', '7kf0', '7vng', '8d47', '7u2d', '6gku', '4g7v', '4ag4', '1xiw', '8ee0', '2zcl', '5wk3', '4xmp', '6xlq', '7r58', '6rcv', '5do2', '6ldw', '3ghe', '5umi', '5w0d', '7sa6', '8dfh', '6iut', '7xnf', '3mac', '5aum', '3opz', '7uqc', '7mlh', '4ers', '4mhh', '1xct', '5xcq', '6q0h', '6pbw', '6ss6', '2h9g', '3wlw', '7n3d', '7ox2', '6ubi', '6aod', '4hs8', '5e8e', '7t72', '7zfe', '7zwm', '5mi0', '1nsn', '5hbt', '3v4v', '6gff', '4dw2', '7dfp', '6j14', '6i8s', '2hkf', '6mfp', '5k59', '5xmh', '5xj4', '3mlw', '6pef', '6xpz', '5tq0', '6bpe', '1rjl', '3o6m', '7mmn', '6whk', '4lmq', '6wo4', '7ekk', '5n7w', '7c4s', '5w1m', '6p67', '3ifo', '6xcj', '4k9e', '2qqn', '4jqi', '3ggw', '7ucx', '4uv7', '6mtn', '6vlw', '6blh', '7ttx', '7trh', '3u0t', '7qf0', '5kw9', '7f7e', '2osl', '6hga', '4ttd', '6d0x', '3kj4', '7so7', '6osh', '4zto', '1jhl', '8gye', '6nmr', '3cxd', '6ncp', '8bsf', '5w6d', '3ld8', '7e88', '5vag', '7jwp', '4i2x', '5kve', '7neg', '7n0v', '7o31', '7lq7', '7mdj', '6a0z', '7q0h', '8dcc', '4idj', '7xsa', '7a0y', '5u3k', '1tzi', '3raj', '5kzp', '7zqt', '7b3o', '4liq', '2ybr', '3whe', '4j6r', '6z7w', '6xkq', '2uzi', '7c88', '6d01', '5vyf', '6bpa', '4hc1', '8szy', '7jwg', '7rqr', '6s5a', '5ggt', '6j5f', '2r56', '4o02', '8bbh', '7skz', '6uvo', '7u09', '6apb', '6uud', '5yy5', '7rda', '7rnj', '6osv', '6r0x', '4r4n', '7oly', '7mzj', '4yzf', '6x1s', '7wue', '6okm', '6xe1', '7x2h', '8ds5', '7syz', '1afv', '8dgx', '5gir', '4lqf', '4ywg', '6q0e', '4fp8', '7dr4', '6phb', '2y06', '6ywc', '6j15', '3x3f', '6bqb', '7wkx', '7tr4', '7e3o', '4xnm', '6db7', '7wph', '6wh9', '3ifn', '4xny', '5myk', '5dmi', '7oxn', '6mto', '7k9z', '8pe9', '7pqz', '3mlt', '5y9j', '4i18']
        

    #    #Antibodies_Nanobodies
    #     self.train_entries = ['7k84', '8c8p', '5hvh', '7zg0', '4w6x', '6ibb', '7n9v', '5o0w', '7b2p', '8h3y', '2p42', '1kxt', '4krm', '7q3q', '6fyt', '5nqw', '7p5w', '6fyu', '7unz', '8cyj', '7r1z', '8ont', '6jb2', '7aqy', '7nbb', '6b73', '7aqg', '7tjc', '7zkw', '7pqg', '7phq', '7a0v', '6c5w', '6cwg', '5m15', '7th2', '3k74', '7w1s', '6u51', '1ri8', '4pir', '7kjh', '2vyr', '4hem', '8gz5', '7r74', '7pa5', '6h71', '7m1h', '5l21', '6o3c', '5jds', '6v7z', '8cy9', '7s2r', '6x07', '6sc7', '4kro', '6f5g', '7q6z', '7d5q', '7p5y', '6i8h', '7fh0', '7rmi', '5vm0', '4hep', '8dtn', '7fat', '8cy6', '7a4t', '7ri1', '7eow', '7p7a', '5vxj', '3p0g', '4krp', '7a4y', '4gft', '6qgx', '5vxm', '7yzi', '4p2c', '7nfq', '7cz0', '1qd0', '5mzv', '6x06', '7qbf', '7r24', '6zg3', '8en4', '4grw', '7ngh', '7b27', '6ul4', '7qvk', '7p16', '7b5g', '4lgs', '7a48', '7nik', '6u12', '6u54', '4z9k', '6rpj', '3k3q', '4i1n', '6zpl', '7sp6', '6oq6', '8cy7', '5tok', '6uht', '7nxx', '7kgk', '7qjj', '4xt1', '7c8v', '4x7f', '4ksd', '6zrv', '1kxq', '5f1k', '8e99', '7z1b', '7n9e', '3k80', '7ozt', '8bev', '8b7w', '6gwp', '7nj5', '6sge', '5mp2', '4qo1', '5e5m', '5vaq', '5toj', '5o05', '7mjh', '6xzu', '7q3r', '8cwu', '3ezj', '7kbi', '7yit', '6fe4', '7r4r', '7wd1', '7fau', '4kfz', '5tjw', '5jqh', '4lgp', '7d2z', '5dfz', '4w2q', '5mje', '7uny', '6gci', '6os1', '8bb7', '4wem', '7k65', '6uft', '4nc2', '7qe5', '5usf', '7zra', '6zwk', '4laj', '7s2s', '4n1h', '7e53', '7jvb', '7jwb', '7d30', '6h6z', '5e7f', '7vfb', '7th3', '8g2y', '6vi4', '6gju', '4w2o', '8f8w', '7n0i', '6h15', '5vxl', '5ihl', '8dqu', '7zmv', '7n9c', '8e0e', '5lwf', '4i0c', '7wn1', '1mvf', '8b41', '7p78', '6rty', '5imm', '6x08', '4orz', '6h02', '6tej', '7upm', '5f1o', '5td8', '6h16', '6obg', '7lvw', '5jmo', '5o03', '6qgw', '7mdw', '7nft', '3v0a', '7me7', '8ew6', '7rga', '6lr7', '2bse', '2x6m', '7nqa', '8en1', '4dk6', '6hhu', '3k81', '6ocd', '7fg3', '7xk9', '8cyc', '7p60', '5ja9', '7o3b', '7wn0', '3stb', '6z1v', '5n88', '5c2u', '7wuj', '6ul6', '7a50', '7aqx', '7kc9', '7wpf', '7z1e', '7sai', '7q6c', '7omm', '6vbg', '5bop', '5f93', '7dv4', '7sp9', '7e6u', '6xw6', '4ios', '1zvy', '6dbf', '5hgg', '7a4d', '8h3x', '6i8g', '5my6', '7usv', '7oap', '6gkd', '5hvf', '6ssp', '7te8', '8cyd', '7om4', '5fv2', '8cxq', '7c8w', '4lhj', '8dtu', '7wki', '6knm', '6qtl', '7my3', '7qbd', '7uia', '7oau', '4yga', '5ovw', '6ey0', '6obc', '7nil', '7t5f', '7o06', '7p6k', '7rby', '1zvh', '4eig', '7d4b', '7kd0', '7na9', '7whj', '5omn', '8dyp', '7czd', '6xw4', '7rnn', '1mel', '5e0q', '6r7t', '8di5', '7km5', '8en5', '8bzy', '7php', '5lhr', '3p9w', '5u4m', '7b2q', '7b2m', '6f2g', '6ssi', '6itq', '8en0', '6ir1', '7a29', '8b01', '7xtp', '6yu8', '7kd2', '7jhg', '7x2j', '5hm1', '7ns6', '6yz7', '3k1k', '4s10', '7z1c', '5e1h', '4weu', '4nc0', '7ubx', '5o02', '7dss', '7pqq', '7now', '7azb', '7ocj', '5m2m', '7x4i', '6hhd', '7r4q', '7bc6', '4w6w', '7nfr', '6ey6', '7nj7', '7ocy', '7f5h', '6n50', '7xqv', '4ocn', '7nmu', '8c3l', '8bf4', '6v7y', '7voa', '6j7w', '6rqm', '6z6v', '7r73', '6s0y', '4eiz', '6x04', '5da0', '7k7y', '7nj3', '7x2m', '7nqk', '5gxb', '7uby', '6ysq', '4c59', '6waq', '6hjy', '5c3l', '4lgr', '7my2', '7z1a', '7qcq', '6zcz', '3g9a', '4mqs', '8a67', '6csy', '6r8x', '2bdn', '5gjt', '5usl', '7fjc', '1kcs', '5e94', '6oz2', '5csz', '3h0t', '8f6o', '6q20', '5u3n', '5t3z', '3pgf', '6mqr', '7l7r', '5eii', '6n5b', '3ztj', '6obd', '1f58', '2zuq', '6u6u', '6x8q', '3hr5', '7o33', '4xgz', '8df1', '2v17', '6cw3', '5bo1', '1cz8', '6bli', '1nmc', '5w06', '7cu4', '2jix', '6oel', '2y6s', '3s36', '5dub', '7jks', '1fbi', '5bjz', '7wvl', '4hcr', '7n4l', '6h06', '6wty', '5vl3', '6z7y', '7kq7', '5mub', '7n3c', '6xm2', '7rt9', '4n8c', '6ldy', '7rxp', '7kmh', '4oqt', '8f60', '5vpl', '6nyq', '3fmg', '7r6x', '1h0d', '8dao', '5njd', '4jzj', '5tzu', '6ayz', '8fax', '7dpm', '8smt', '8ahn', '5v6m', '7t25', '7uvf', '7xsb', '5f3h', '5tkk', '3sge', '4g80', '7orb', '4kvn', '5jhl', '6p60', '5th9', '4xnz', '3vrl', '6bfq', '4rgn', '5myo', '7fcp', '3b2u', '7wg3', '5wna', '5kvg', '7ul0', '6wf0', '5u3j', '6xli', '2h1p', '6qig', '7e72', '6wg0', '7t86', '5vic', '5tkj', '4f15', '5a3i', '6t9e', '6vry', '7chf', '6lgw', '5x0t', '6aru', '7kpb', '8dcy', '4dke', '5sx4', '8gb5', '7cho', '5uoe', '5umn', '6gk8', '7q4q', '7ufo', '5xhv', '6mtq', '3ncy', '2hh0', '3sqo', '7tuf', '7o9s', '6o2b', '4aei', '4z5r', '5wko', '8ath', '6nnj', '6cbv', '7ue9', '7sg6', '7pi2', '7rah', '3nps', '1qle', '6m38', '3vi3', '7jtg', '8dn6', '6pxr', '8d36', '7tp3', '6ii4', '7lf8', '7u8c', '7s0x', '7ssc', '6x78', '8cwj', '6dfj', '5t33', '5vk2', '5fgc', '4uu9', '7ce2', '5w08', '7beh', '8dgu', '4gxu', '8eoo', '4jpv', '5utz', '7txw', '7yds', '5w3p', '7cgw', '3pnw', '8db4', '4am0', '6b0e', '7unb', '7s4s', '6x7w', '7dk0', '7ps7', '6oej', '8d6z', '7u8e', '2eh8', '7e5y', '3so3', '6n7j', '3i50', '4glr', '7zxk', '2qqk', '4g6a', '4u6h', '5f9w', '8f6l', '7k75', '5czx', '6umx', '6xpy', '7y3j', '4o58', '5ywy', '7s0b', '7x8t', '5nh3', '5ukr', '7r8u', '4xvt', '7mzl', '7pry', '7z2m', '4rx4', '7cjf', '7wbz', '7kql', '5cws', '6o9i', '6p8n', '3iet', '7x8q', '5tlj', '5tpn', '7shy', '6bzy', '6tyb', '5te6', '4q6i', '7l2c', '4jb9', '4n9g', '7mfr', '7phu', '4v1d', '6iap', '7cdj', '7kpg', '7xjf', '3sdy', '6id4', '7ox1', '8hit', '4m8q', '7djz', '7kmg', '6dkj', '8gng', '5b3j', '3nh7', '7tp4', '4xvu', '6wfx', '5dmg', '4tsa', '6flb', '7lr3', '5occ', '5lqb', '6vbo', '3ab0', '6oy4', '1sy6', '4hg4', '8d9y', '4zs6', '4s1r', '5kaq', '2fjh', '5tbd', '7e5o', '6vc9', '6k0y', '6mqm', '6o9h', '6o41', '7zfd', '7s5q', '7kmi', '1jps', '6mqe', '6xp6', '4zpv', '4j8r', '4jzn', '7mmo', '1oak', '2jel', '5dfv', '8dxt', '7or9', '8ib1', '6zvf', '8dn7', '5w5z', '5wux', '5x8l', '4q0x', '4rav', '6w16', '1u8p', '7rew', '6gv1', '7ps4', '1zea', '1kc5', '7beo', '4hpy', '3tt3', '6b0h', '5y2l', '1ktr', '6bkd', '5d1q', '8ol9', '7e9b', '4ydi', '8fan', '1egj', '8euq', '6ii9', '3c09', '8gq1', '8as0', '4ydv', '4s1s', '7nks', '7bbj', '7jmo', '6wo3', '6w4m', '1ob1', '1eo8', '6t3j', '7kfv', '7o30', '7lcv', '6wg1', '1orq', '7ahv', '1a3r', '5myx', '8fdo', '6uyg', '7mzf', '4rwy', '6bdz', '5l0q', '6fy3', '8dtr', '5kan', '3r1g', '3mlz', '7sue', '7prz', '6plh', '6aq7', '7rd5', '1xgy', '8djg', '7s8h', '6e56', '4l5f', '7raq', '5udc', '6ct7', '2xra', '8gpy', '4yqx', '3n85', '6n81', '2vxq', '6pdu', '5uem', '4hf5', '8hpk', '5d1x', '6wqo', '8gho', '4zso', '6cxy', '6wvz', '6pdx', '6ule', '7vzt', '7zfb', '8bbo', '6j6y', '5fgb', '4o9h', '4tul', '7re7', '2vwe', '6xlz', '6h3u', '8fa9', '6pcu', '6yxd', '7uvs', '4rfo', '6d11', '8hpj', '5eu7', '6bzw', '7lr4', '6rps', '3be1', '7lfb', '4n0y', '6iea', '6hhc', '5dur', '6o3b', '7si2', '2j5l', '5w23', '7ps2', '7rfb', '6ieb', '6iek', '6hxw', '7ucf', '6x1w', '5i9q', '7st5', '6x1v', '5w6g', '4yfl', '7e86', '7ean', '6a67', '6xq0', '5ikc', '6b0g', '8gpu', '7yv1', '1qfu', '6vgr', '4kuc', '7x63', '5ush', '5d8j', '6sf6', '1ndm', '6df2', '7t7b', '6ewb', '3thm', '3uc0', '4f2m', '7xeg', '6b3m', '7dk2', '2h32', '5ocx', '6ous', '7tri', '8ag1', '6x1u', '2nr6', '4yhz', '6db6', '6xzw', '4qhu', '6mns', '7vyt', '4edx', '7bxv', '3tt1', '7s13', '4xi5', '4om0', '4uao', '6mqc', '4ht1', '5hdq', '7kn4', '7fah', '7shu', '7yck', '6j11', '4rgm', '1fj1', '1e4x', '5w1k', '7mzk', '1uwx', '6pzf', '1vfb', '6wfy', '8elo', '2q8a', '3kr3', '6xxv', '7ps5', '3ks0', '7y8j', '1yqv', '5vqm', '8fdc', '4nnp', '7x9e', '1bj1', '6q18', '6u36', '2aep', '7vyr', '7ul1', '6fy1', '7z0y', '5lbs', '6woz', '5dum', '8eee', '2qr0', '7qu2', '1ai1', '7sfx', '8gb7', '6n5d', '6dc5', '6nha', '6vmj', '4dn4', '5if0', '6ieq', '4wuu', '7zf3', '7jmp', '5t6p', '8eb2', '8byu', '4u1g', '6cyf', '6yxk', '5hj3', '7jix', '5drz', '4ffy', '6eay', '4y5v', '1kcr', '3ubx', '6mlk', '7u0e', '8aon', '7chz', '6ion', '1ynt', '6dc3', '3qnz', '8gf2', '4h88', '6dca', '5x08', '7e7e', '7kf1', '1ors', '5o1r', '7dha', '6mvl', '1yjd', '6o39', '6xpx', '1v7n', '8dk6', '4hfu', '7u0a', '5veb', '7df1', '5tz2', '5epm', '5tlk', '2r0l', '6p3s', '3liz', '8fba', '6txz', '8bbn', '7tug', '5czv', '4jhw', '6cmg', '7coe', '7te1', '1xgq', '6p9h', '6vy4', '8dnn', '6pec', '6f7t', '7pa9', '7yru', '1z3g', '6cse', '8av9', '1ggi', '5jz7', '6vjt', '7v64', '7bh8', '5y11', '8j7e', '5tih', '6lxj', '7seg', '6pds', '6p8m', '5u7o', '4mwf', '7c94', '7cj2', '6d2p', '2g5b', '4bkl', '5cjx', '4k2u', '5h35', '4iof', '6n5e', '7uzd', '7qnw', '5n0a', '4jdt', '4zff', '7uxl', '6vi1', '6mi2', '5lsp', '5uqy', '4cad', '6h3t', '5esv', '7wnb', '5vcn', '7pi7', '8gh4', '4d9r', '6svl', '8f0i', '6glx', '7rd3', '1ahw', '6yxm', '7ps0', '6vbq', '6u9s', '6wit', '6vep', '6x4t', '6o3a', '5dsc', '1hez', '7lsf', '6bit', '7lja', '3g04', '5eoq', '3b9k', '7sts', '6kva', '3qg6', '8dtt', '7dm1', '6lyn', '4xxd', '2gsi', '4m5z', '1ztx', '5c0n', '7rdw', '5dlm', '5ggv', '6dc9', '3dvn', '7n0x', '7ufn', '7sqw', '6o3l', '8dfi', '3ulu', '5ob5', '7m7w', '5mhr', '6cez', '4dag', '5grj', '6att', '6jbt', '6i3z', '6xsw', '7bnv', '7xsc', '4n90', '6ba5', '8hn7', '4ubd', '6fgb', '2w9e', '6vvu', '2ny3', '3l5y', '7z0x', '7m3i', '6uoe', '7w71', '7jie', '5nj6', '6cdp', '6awo', '7eam', '5wb9', '7uvh', '4yk4', '7t87', '6wmw', '7jtf', '6w05', '8gy5', '8caf', '1mvu', '7pi3', '7x8p', '5k9o', '7amr', '5w4l', '1ezv', '6oz9', '6vca', '5tzt', '6kz0', '6o24', '8dwa', '6wir', '7ycn', '7yzj', '6pi7', '7bel', '7bbg', '5dup', '7ceb', '6db5', '6k65', '3kj6', '7lki', '8slb', '7fjs', '6vbp', '2j88', '7uvq', '1nfd', '4rrp', '7rfc', '7sjs', '6snc', '1nak', '1ndg', '5ifj', '5o4o', '6xq2', '8b9v', '4jre', '8h8q', '6vug', '5zia', '7y0c', '2r29', '8gb6', '7chs', '7ytn', '7uzc', '7x7o', '6wj1', '8c3v', '4yr6', '6wm9', '5mhs', '6o25', '4hj0', '6w51', '4bz2', '3hae', '7qu1', '5cbe', '8f0l', '4k3j', '7ukt', '6lz4', '6pbv', '8bse', '7xrz', '6p7h', '4jan', '5v6l', '7ox3', '6bf4', '7kyo', '5wtt', '6cwt', '3eff', '3v4u', '5ijk', '1ikf', '8ekf', '6w7s', '6hf1', '6yio', '5c6t', '6pis', '7pa8', '3se8', '6a4k', '7njz', '3hmx', '7zf8', '8dxu', '4jpw', '1adq', '4y5x', '6m58', '4edw', '7keo', '6h5n', '8e1g', '4cni', '6mnq', '7tcq', '7cr5', '7wsl', '8f5i', '6wzk', '6b0n', '6lz9', '2oz4', '6bzu', '7lfa', '6pdr', '6icc', '4leo', '7xik', '7quh', '7joo', '6xr0', '6ddv', '5ggr', '5ea0', '6i9i', '7uvi', '1e6j', '5cus', '8dcm', '7s4g', '8oxv', '8ee1', '1fsk', '6ktr', '7l7e', '5ldn', '7mdp', '7ycl', '6i04', '7np1', '5o4g', '4qww', '3wsq', '7pgb', '5vob', '7s7i', '5eoc', '7rxi', '6xqw', '4ypg', '5u8q', '4ydl', '4o4y', '5kvd', '5gzo', '1lk3', '8ivx', '7bq5', '6bzv', '5o14', '6nmt', '6phf', '6urh', '4wht', '7msq', '4oii', '2xqb', '4u6v', '3qum', '8ddk', '7klg', '1jrh', '4zfo', '7mzm', '6mqs', '1fpt', '3lqa', '7k66', '6jep', '4dkf', '6wno', '8d0a', '6a77', '6ii8', '3u9u', '8dfg', '5d1z', '7cm4', '2brr', '5vxr', '5y9c', '5kvf', '5fec', '7sem', '4hzl', '6was', '3gi8', '6iuv', '3h42', '7n0u', '3wkm', '7sl5', '6ddr', '6cw2', '8gpx', '2r0z', '4lkx', '6e4y', '6wtv', '8ee8', '6wo5', '5c8j', '6otc', '4hs6', '5vl7', '6x1t', '6elu', '4mxw', '3g6j', '2r0k', '4r3s', '5nmv', '7rm0', '7l5b', '8dw9', '7n4j', '5d72', '7ahu', '7y0v', '7rks', '7ejz', '3ujj', '6z06', '7e7y', '5mvz', '8ee5', '8e1p', '7wn2', '6iw0', '3idy', '1hh6', '6n8d', '3w2d', '7dc8', '6uig', '7rku', '4od2', '6q23', '6ln2', '7u2e', '4i3r', '8f5n', '3skj', '6azz', '8ivw', '6xkr', '8hn6', '2qhr', '6n16', '7mzg', '8fdd', '7yxu', '6ohg', '5c0r', '5gmq', '5wdf', '4i77', '5tud', '6uta', '6bb4', '4xmk', '5ye3', '6ppg', '4okv', '6vy5', '1qkz', '4ut6', '7kqh', '6erx', '4tnw', '3fn0', '5xwd', '6mtt', '7u5b', '7vux', '6adc', '7so5', '8f7z', '3u9p', '2hfg', '3v6o', '5zs0', '7uvo', '6bck', '5b71', '6ute', '7m51', '3efd', '7s07', '3qa3', '6jjp', '8ffe', '6wg2', '2a6k', '1qfw', '8be1', '6q1z', '8f9v', '6e3h', '4qti', '7n08', '5vlp', '5xbm', '6e4x', '7amq', '6phd', '6nmu', '6m3c', '7s11', '3bgf', '4p59', '6hx4', '6r2s', '5yoy', '6axl', '6o1f', '8dgv', '3lzf', '2zjs', '7o2z', '3nfp', '4r0l', '1tzh', '4rau', '2qsc', '3d85', '7vtj', '6p95', '4m1g', '7q0i', '6nmv', '7zmk', '6wx2', '7kyl', '1kb5', '5eor', '6xq4', '6uyn', '8fg0', '8fa6', '5mo9', '6plk', '8dgw', '5dwu', '3go1', '1ifh', '3eob', '3bae', '7kn3', '4ydj', '8eed', '6co3', '5vkd', '8f95', '3cvh', '4np4', '6rlo', '6o26', '4yue', '2uud', '7sbu', '5ibl', '4onf', '7rxj', '2j6e', '5y9f', '5x2o', '7rqq', '4lu5', '4plk', '7yk4', '4y5y', '7t0l', '8fas', '7mf1', '5bk2', '4dqo', '3jwo', '6mtp', '6d9w', '5t5n', '4j4p', '7bep', '7um3', '6h2y', '3e8u', '6wjl', '6ye3', '6xy2', '5usi', '7ly3', '7xy8', '7e7x', '2xtj', '6flc', '1za3', '7mzn', '6ldv', '7sx7', '3bky', '7wn8', '3zkn', '5ug0', '4ybq', '7n0a', '7bsc', '2nyy', '7t82', '7lm8', '5kn5', '6pze', '7st8', '6z2l', '4qnp', '1ken', '8f9u', '3o0r', '4ye4', '6hig', '3p0y', '6glw', '8dtx', '7ew5', '7n4i', '2i9l', '4zs7', '6bp2', '7qny', '4tsb', '3uji', '6cdm', '3lex', '4jg0', '6o23', '5wi9', '6ddm', '7rxl', '4bz1', '6nms', '6iec', '8el2', '8duz', '7r6w', '2arj', '5anm', '1nca', '6m3b', '5bk1', '4ut9', '5xcu', '6mg7', '5gzn', '1g9m', '7k8m', '1pkq', '3g5y', '7doh', '4xh2', '6pe9', '6yax', '8fat', '8dv6', '7rlw', '3t2n', '4yhp', '6c6z', '3u30', '1ejo', '6k7o', '3v7a', '8a44', '7s5p', '7lk4', '5vig', '4xvj', '7kqg']
    #     self.valid_entries = ['6eqi', '7vq0', '7qiv', '7paf', '8bpk', '7klw', '8dfl', '7mfu', '7ust', '8cyb', '7l6v', '7zw1', '7kdu', '6uc6', '7nk4', '8emy', '7ph4', '7kbk', '6obm', '7yag', '4lhq', '7sla', '6ze1', '6z20', '8gni', '7can', '5h8o', '5ivn', '1zv5', '5boz', '8emz', '4y7m', '7s7r', '5m2j', '7tpr', '4x7d', '4pgj', '3cfi', '4fhb', '5vl2', '4bfb', '5vnw', '6app', '4pou', '3jbc', '7nx0', '6qpg', '7n9t', '5vm6', '6gjq', '7z85', '5vxk', '5ja8', '7vfa', '6zxn', '4i13', '8cxn', '5fhx', '7kji', '4wen', '4hjj', '8en3', '3rjq', '7z7x', '3qsk', '2p4a', '7n0r', '6obo', '7nki', '7y9t', '5nbl', '4cdg', '1op9', '6qv1', '6oq7', '6lz2', '5f7l', '8sbb', '6xw5', '7ar0', '7sl9', '6qv2', '6i6j', '6quz', '1kxv', '7pc0', '6oca', '5lhp', '6tyl', '7kkk', '6x05', '8cxr', '7x7e', '7zk1', '7vke', '6o8d', '7x2l', '7o31', '4w6y', '3k7u', '6qx4', '5f21', '7r4i', '7qbg', '7dst', '7saj', '6v80', '5j1t', '7xrp', '5ip4', '7epb', '5m30', '4m3k', '6urm', '2b1h', '6wn1', '6vjn', '6p8d', '3ifp', '6b0a', '7b0b', '7str', '6d0u', '7q6c', '4m1d', '7mnl', '5w0k', '4xak', '7s3m', '7d85', '6mhr', '7s1b', '8cz5', '7eyc', '7kf0', '7vng', '8d47', '7u2d', '6gku', '4g7v', '4ag4', '1xiw', '8ee0', '2zcl', '5wk3', '4xmp', '6xlq', '7r58', '6rcv', '5do2', '6ldw', '3ghe', '5umi', '5w0d', '7sa6', '8dfh', '6iut', '7xnf', '3mac', '5aum', '3opz', '7uqc', '7mlh', '4ers', '4mhh', '1xct', '5xcq', '6q0h', '6pbw', '6ss6', '2h9g', '3wlw', '7n3d', '7ox2', '6ubi', '6aod', '4hs8', '5e8e', '7t72', '7zfe', '7zwm', '5mi0', '1nsn', '5hbt', '3v4v', '6gff', '4dw2', '7dfp', '6j14', '6i8s', '2hkf', '6mfp', '5k59', '5xmh', '5xj4', '3mlw', '6pef', '6xpz', '5tq0', '6bpe', '1rjl', '3o6m', '7mmn', '6whk', '4lmq', '6wo4', '7ekk', '5n7w', '7c4s', '5w1m', '6p67', '3ifo', '6xcj', '4k9e', '2qqn', '4jqi', '3ggw', '7ucx', '4uv7', '6mtn', '6vlw', '6blh', '7ttx', '7trh', '3u0t', '7qf0', '5kw9', '7f7e', '2osl', '6hga', '4ttd', '6d0x', '3kj4', '7so7', '6osh', '4zto', '1jhl', '8gye', '6nmr', '3cxd', '6ncp', '8bsf', '5w6d', '3ld8', '7e88', '5vag', '7jwp', '4i2x', '5kve', '7neg', '7n0v', '7o31', '7lq7', '7mdj', '6a0z', '7q0h', '8dcc', '4idj', '7xsa', '7a0y', '5u3k', '1tzi', '3raj', '5kzp', '7zqt', '7b3o', '4liq', '2ybr', '3whe', '4j6r', '6z7w', '6xkq', '2uzi', '7c88', '6d01', '5vyf', '6bpa', '4hc1', '8szy', '7jwg', '7rqr', '6s5a', '5ggt', '6j5f', '2r56', '4o02', '8bbh', '7skz', '6uvo', '7u09', '6apb', '6uud', '5yy5', '7rda', '7rnj', '6osv', '6r0x', '4r4n', '7oly', '7mzj', '4yzf', '6x1s', '7wue', '6okm', '6xe1', '7x2h', '8ds5', '7syz', '1afv', '8dgx', '5gir', '4lqf', '4ywg', '6q0e', '4fp8', '7dr4', '6phb', '2y06', '6ywc', '6j15', '3x3f', '6bqb', '7wkx', '7tr4', '7e3o', '4xnm', '6db7', '7wph', '6wh9', '3ifn', '4xny', '5myk', '5dmi', '7oxn', '6mto', '7k9z', '8pe9', '7pqz', '3mlt', '5y9j', '4i18']
    #     self.test_entries = ['6qup', '7qbe', '7f5g', '7nkt', '7b14', '7olz', '7zfb', '7wd2', '6fyw', '8dlx', '5nbd', '6b20', '7b17', '7lzp', '6u55', '5tsj', '7p2d', '8en2', '7ldj', '6qgy', '1g6v', '6xw7', '6oq5', '3jbd', '5m14', '6fv0', '6i2g', '5m2i', '6obe', '5eul', '6war', '4nc1', '6ruv', '2xv6', '6nfj', '7kkl', '5g5x', '6rvc', '7sqp', '5fuc', '5sv3', '7aej', '1rjc', '7apj', '7aqz', '8en6', '6c9w', '6h6y', '6gs1', '7nvm', '5y80', '6q6z', '7tgf', '7tgi', '2x89', '5mwn', '7qne', '3ffd', '4etq', '4ki5', '2cmr', '4qci', '1iqd', '1uj3', '5en2', '5nuz', '2ypv', '4cmh', '2b2x', '1osp', '3k2u', '2xwt', '4fqj', '3rkd', '2adf', '4ot1', '5l6y', '3bn9', '3hi6', '3l95', '3s35', '2xqy', '3o2d', '5f9o', '5j13', '5ggs', '4ffv', '4ydk', '1fe8', '4lvn', '4dtg', '5bv7', '1w72', '5d96', '4jlr', '7dm2', '8fbw', '6vel', '7qf1', '6nz7', '6oe4', '4wv1', '6bfs', '5k9q', '4jo1', '7sbd', '3p30', '8cyh', '1i9r', '6xkp', '5tfw', '8fa7', '6kyz', '1frg', '7jkt', '6oor', '5t1d', '5jq6', '7mzh', '7chp', '6t3f', '3iu3', '6mek', '5xku', '3bsz', '5xj3', '2vis', '7pr0', '6n6b', '6b5r', '4p3c', '4d3c', '7kez', '5fb8']
