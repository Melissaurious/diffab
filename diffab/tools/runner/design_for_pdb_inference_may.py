import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/rioszemm/data/NanobodiesProject/diffab/')

print("Current working directory:", os.getcwd())
print("Python sys.path:", sys.path)
print(sys.executable)


from diffab.datasets.custom import preprocess_antibody_structure
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
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
    # Load configs
    # print("args.config", args.config)
    # config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else 2022)# according to config json files 

    # Structure loading
    # data_id = os.path.basename(args.pdb_path)
    data_id = args.pdb

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

    def get_model(cfg):
        return _MODEL_DICT[cfg.type](cfg)
        
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

                # save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                # save_path = os.path.join(log_dir, '%04d.pdb' % (count, ))
                save_path = os.path.join(log_dir, args.pdb_code + ".pdb")
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
                with open(args.summary_dir, 'a') as f:
                    summary ={
                    "mod_pdb": save_path,
                    # "ref_pdb": os.path.join(log_dir, 'REF1.pdb'),
                    "ref_pdb": args.pdb_path,
                    'heavy_chain':args.heavy,
                    'light_chain': args.light,
                    'antigen_chains': args.antigen,
                    'cdr_type': ["H3"],
                    'pdb':args.pdb_code,
                    # 'model': args.model
                    }
                    f.write(json.dumps(summary) + '\n')

                count += 1

        logger.info('Finished.\n')

    # return summary


def main(): 
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained checkpoint')
    parser.add_argument('--ckpt_folder', type=str, required=True, help='Path to the folder of trained checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the results')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--summary_file', type=str, required=True, help='where to save summary')
    parser.add_argument('--FOLD', type=int, default=None)

    parser.add_argument('--no_renumber', action='store_true', default=True) # Do not do renumbering
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=1)
    args = parser.parse_args()


    with open(args.test_set, 'r') as fin:
        data = fin.read().strip().split('\n')

    checkpoint = os.listdir(args.ckpt_folder)[0]
    checkpoint_full_path = os.path.join(args.ckpt_folder,checkpoint)

    # write down the summary
    summary_file = args.summary_file
    # summary_file = os.path.join(args.summary_dir, f'summary_fold_{args.FOLD}.json')
    if not os.path.exists(summary_file):
        with open(summary_file, 'w') as fout:
            # for item in summary_items:
            #     fout.write(json.dumps(item) + '\n')
            pass


    # Iterate over each entry in the JSON file
    # summary_items = []
    for entry in data:
        json_obj = json.loads(entry)
        pdb_path= json_obj['pdb_data_path']



        # pdb_code=json_obj['pdb']
        pdb_unique_code = json_obj["entry_id"]
        save_path = os.path.join(args.out_dir, pdb_unique_code + ".pdb")
        if os.path.exists(save_path):
            continue

        args = EasyDict(
            # pdb_path=json_obj['pdb_data_path'], # this is in imgt numbering, change to the chothia folder.
            pdb_path=pdb_path,
            pdb=json_obj['pdb'],
            heavy=json_obj['heavy_chain'],
            light=json_obj['light_chain'],
            antigen=json_obj['antigen_chains'],
            num_samples=1,
            no_renumber =args.no_renumber,
            seed=args.seed,
            out_dir=args.out_dir,
            checkpoint=checkpoint_full_path,
            device=args.device,
            batch_size=args.batch_size,
            tag=args.tag,
            pdb_code=pdb_unique_code,
            summary_dir=summary_file
        )

        try:
            design_for_pdb(args)
        except Exception as e:
            error_message = f"Error processing {json_obj['entry_id']}: {e}"
            print(error_message)
            continue


if __name__ == '__main__':
    main()