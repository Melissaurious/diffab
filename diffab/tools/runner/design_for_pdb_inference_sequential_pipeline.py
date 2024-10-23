import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import os
import json
import argparse
from tqdm import tqdm
from shutil import rmtree
import logging

logging.disable('INFO')
import glob
import sys
import sys
sys.path.append('/home/rioszemm/data/NanobodiesProject/diffab/')

# print("Current working directory:", os.getcwd())
# print("Python sys.path:", sys.path)
# print(sys.executable)


from diffab.datasets.custom import preprocess_antibody_structure
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.models import get_model
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *
# from diffab.models import get_model
import time

# from diffab.tools.renumber import renumber as renumber_antibody



def create_data_variants(structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    cdrs = sorted(list(set(find_cdrs(structure)).intersection(["H_CDR3"])))
    for cdr_name in cdrs:
        transform = Compose([
            MaskSingleCDR(cdr_name, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        residue_first, residue_last = get_residue_first_last(data_var)
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-{cdr_name}',
            'tag': f'{cdr_name}',
            'cdr': cdr_name,
            'residue_first': residue_first,
            'residue_last': residue_last,
        })

    return data_variants


def design_for_pdb(args):

    seed_all(args.seed if args.seed is not None else 2022)# according to config json files 

    # Structure loading
    # data_id = os.path.basename(args.pdb_path)
    data_id = args.entry_id

    if args.no_renumber:
        pdb_path = args.pdb_path
    else:
        in_pdb_path = args.pdb_path
        out_pdb_path = os.path.splitext(in_pdb_path)[0] + '_chothia.pdb'
        heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
        pdb_path = out_pdb_path

        if args.heavy is None and len(heavy_chains) > 0:
            args.heavy = heavy_chains[0]
        if args.light is None and len(light_chains) > 0:
            args.light = light_chains[0]
            
    if args.heavy is None and args.light is None:
        raise ValueError("Neither heavy chain id (--heavy) or light chain id (--light) is specified.")
    get_structure = lambda: preprocess_antibody_structure({
        'id': data_id,
        'pdb_path': pdb_path,
        'heavy_id': args.heavy,
        # If the input is a nanobody, the light chain will be ignored
        'light_id': args.light,
    })

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    # tag_postfix = '_%s' % args.tag if args.tag else ''
    # log_dir = get_new_log_dir(
    #     os.path.join(args.out_root, config_name + tag_postfix), 
    #     prefix=data_id
    # )
    log_dir = args.out_dir
    # os.makedirs(log_dir)
    logger = get_logger('sample', log_dir)
    logger.info(f'Data ID: {structure_["id"]}')
    logger.info(f'Results will be saved to {log_dir}')
    data_native = MergeChains()(structure_)
    # save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (args.checkpoint))
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']

        
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    # Make data variants
    data_variants = create_data_variants(
        # config = config,
        structure_factory = get_structure,
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': data_id,
        # 'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    # if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
    #     inference_tfm.append(RemoveNative(
    #         remove_structure = config.sampling.sample_structure,
    #         remove_sequence = config.sampling.sample_sequence,
    #     ))
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        # os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        os.makedirs(os.path.join(log_dir), exist_ok=True)

        logger.info(f"Start sampling for: {variant['tag']}")
        
        # save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
        # save_pdb(data_native, os.path.join(log_dir, 'REF1.pdb'))       # w/  OpenMM minimization

        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )
        data_list_repeat = [ data_cropped ] * args.num_samples   #config.sampling.num_samples
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        count = 0
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            # if 'abopt' in config.mode:
            #     # Antibody optimization starting from native
            #     traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
            #         'pbar': True,
            #         'sample_structure': config.sampling.sample_structure,
            #         'sample_sequence': config.sampling.sample_sequence,
            #     })
            # else:
            # De novo design
            traj_batch = model.sample(batch, sample_opt={
                'pbar': True,
                'sample_structure': True,
                'sample_sequence': True,
            })

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )


                save_path = os.path.join(log_dir, data_id, f"{data_id}_{count}.pdb")
                try:
                    # save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                    # save_path = os.path.join(log_dir, '%04d.pdb' % (count, ))
                    save_pdb({
                        'chain_nb': data_tmpl['chain_nb'],
                        'chain_id': data_tmpl['chain_id'],
                        'resseq': data_tmpl['resseq'],
                        'icode': data_tmpl['icode'],
                        # Generated
                        'aa': aa,
                        'mask_heavyatom': mask_ha,
                        'pos_heavyatom': pos_ha,
                    }, path=save_path)

                    import sys 
                    sys.path.append('/home/rioszemm/data/NanobodiesProject/dyMEAN')
                    from utils.renumber import renumber_pdb
                    renumber_pdb(save_path, save_path, scheme = "imgt")

                    pdb_code = data_id.rsplit('_')[0]
                    with open(args.summary_dir, 'a') as f:
                        summary ={
                        "pdb": pdb_code,
                        "mod_pdb": save_path,
                        # "ref_pdb": os.path.join(log_dir, 'REF1.pdb'),
                        "ref_pdb": args.ref_pdb,
                        'heavy_chain':args.heavy,
                        'light_chain': args.light,
                        'antigen_chains': args.antigen,
                        'cdr_type': ["H3"],
                        'entry_id':data_id,
                        # 'model': args.model
                        }
                        f.write(json.dumps(summary) + '\n')

                    count += 1

                except Exception as e:
                    print(f"Error processing file: {e}")
                    # Check if the file exists before attempting to delete it
                    if os.path.exists(save_path):
                        os.remove(save_path)  # Delete the problematic file
                    continue  # Proceed with the next batch
                    

        logger.info('Finished.\n')

    # return summary


class Arg:
    def __init__(self, pdb_path, heavy, light, antigen, out_dir, summary_dir,cdr_type, num_samples, checkpoint, entry_id,ref_pdb):
        self.pdb_path = pdb_path
        self.heavy = heavy
        self.light = light
        self.antigen = antigen
        self.no_renumber = True
        self.out_dir = out_dir
        self.tag = ''
        self.seed = 0
        self.device = 'cuda'
        self.batch_size = 16
        self.summary_dir = summary_dir
        self.cdr_type = cdr_type
        self.num_samples = num_samples
        self.checkpoint = checkpoint
        self.entry_id = entry_id
        self.ref_pdb = ref_pdb


def main(args):

    with open(args.dataset, 'r') as fin:
        data = [json.loads(line) for line in fin]

    print(len(data))

    # create a dictionary of {pdb:whole dictionary correspinding to such pdb}
    pdb_dict = {}
    for item in data:
        entry_id = item.get("entry_id", item.get("pdb"))
        pdb_dict[entry_id] = item

    # print(pdb_dict)
    print(pdb_dict.keys())
    pdb_entry_ids = list(pdb_dict.keys())
    # print("pdb_entry_ids", pdb_entry_ids)

    if not os.path.exists(args.out_dir):  # do i want everything in the same folder or shall i create a folder per Pdb? still deciding if producing 1 or more
        os.makedirs(args.out_dir)

    summary_dir = os.path.join(args.out_dir, f"summary.json")
    if not os.path.exists(summary_dir):
        with open(summary_dir, 'w') as f:
            pass

    # directories = os.listdir(args.hdock_models)
    # directories = [directory for directory in directories if os.path.isdir(os.path.join(args.hdock_models, directory))]
    # directories = []

    # pdb_paths = []
    # for entry in pdb_entry_ids:
    #     full_path = os.path.join(args.hdock_models,entry, entry + ".pdb")
    #     pdb_paths.append(full_path)

    for entry in pdb_entry_ids:

        print("subdir", entry)
        # print("fullpath", full_path) # /home/rioszemm/data/april_2024_DIFFAB_nano_clst_Ag_sequential_pipeline/fold_5/HDOCK_1/7qn7_F_X_C

        # hdock_models = [os.path.join(full_pat, file) for file in os.listdir(full_path) if file.endswith('.pdb') and if file.startswith('.pdb')] # @ hdock models all the top docked for all entries will be in there
        # hdock_models = [os.path.join(full_path, file) for file in os.listdir(full_path) if file.endswith('.pdb') and file.startswith('model')]

        # for hdock_model in hdock_models:
        # entry_id = hdock_model.split("/")[-1].split(".")[0]

        ouput_dir = os.path.join(args.out_dir, entry)
        print("ouput_dir", ouput_dir)
        # print(ouput_dir) 

        if not os.path.exists(ouput_dir):
            os.makedirs(ouput_dir)

        # Check if output directory exists and contains 100 PDB files
        if os.path.exists(ouput_dir) and len(os.listdir(ouput_dir)) >= int(args.num_samples): 
            print(f"Skipping {ouput_dir} as it already contains designs.")
            continue  # Skip to next hdock_model

        full_path = os.path.join(args.hdock_models,  entry + ".pdb")
        # full_path = os.path.join(args.hdock_models,entry, entry + ".pdb")
        item = pdb_dict.get(entry)
        hdock_model = full_path
        import sys
        sys.path.append('/home/rioszemm/data/NanobodiesProject/dyMEAN')
        from utils.renumber import renumber_pdb
        try:
            renumber_pdb(hdock_model, hdock_model, scheme="chothia")
        except Exception as e:
            print(f"An error occurred during the renumbering process: {str(e)}")

    
        item = pdb_dict.get(entry)
        print(item)
        ref_pdb = item["pdb_data_path"]
        H, L, A = item["heavy_chain"], item["light_chain"], item["antigen_chains"]
        entry_id = entry
        sys.path.append('/home/rioszemm/data/NanobodiesProject/diffab')
        design_for_pdb(Arg(hdock_model, H, L, A, args.out_dir, summary_dir ,args.cdr_type, args.num_samples, args.checkpoint, entry_id,ref_pdb))



def parse():
    parser = argparse.ArgumentParser(description='generation by diffab')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--hdock_models', type=str, default=None, help='Hdock directory')
    parser.add_argument('--cdr_type', type=str, default='H3', help='Type of CDR',
                        choices=['H3'])
    parser.add_argument('--checkpoint', type=str, default=None, help='GPU')
    parser.add_argument('--num_samples', type=int, default=None, help='GPU')
    parser.add_argument('--gpu', type=int, default=None, help='GPU')


    return parser.parse_args()


if __name__ == '__main__':
    
    main(parse())
