import os
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import sys
import csv
import pandas as pd
from tqdm import tqdm

from retroformer.utils.build_utils import build_model, build_iterator, load_checkpoint, accumulate_batch
from retroformer.utils.model_utils import validate
from retroformer.utils.loss_utils import LabelSmoothingLoss

class Args:
    def __init__(self):
        self.device = 'cuda'
        self.batch_size_trn = 2
        self.batch_size_val = 2
        self.batch_size_token = 4096
        self.data_dir = None
        self.intermediate_dir = None
        self.checkpoint_dir = None
        self.checkpoint = 'biochem.pt'
        self.encoder_num_layers = 8
        self.decoder_num_layers = 8
        self.d_model = 256
        self.heads = 8
        self.d_ff = 2048
        self.dropout = 0.3
        self.known_class = 'False'
        self.shared_vocab = 'True'
        self.shared_encoder = 'False'
        self.max_epoch = None
        self.max_step = 3000000
        self.report_per_step = 200
        self.save_per_step = 2500
        self.val_per_step = 10000
        self.verbose = 'False'

args = Args()

def anneal_prob(step, k=2, total=150000):
    step = np.clip(step, 0, total)
    min_, max_ = 1, np.exp(k * 1)
    return (np.exp(k * step / total) - min_) / (max_ - min_)

def main(args):

    validation_accuracy_overtime = pd.DataFrame()

    if not os.path.exists('/log'):
        log_dir = os.path.join(args.intermediate_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
    log_file_name = 'log/' + datetime.now().strftime("%D:%H:%M:%S").replace('/', ':') + '.txt'
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    with open(log_file_name, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    train_iter, val_iter, vocab_itos_src, vocab_itos_tgt = build_iterator(args, train=True, sample=False, augment=True)
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    global_step = 1
    if args.checkpoint:
        global_step, _, model = load_checkpoint(args, model)
        global_step += 1

    learning_rate = None
    if args.lowerlr:
        learning_rate = 0.00001 if args.times == 10 else 0.000001 if args.times == 100 else 0.0000001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    criterion_bond_rc = torch.nn.BCELoss(reduction='sum')
    criterion_atom_rc = torch.nn.BCELoss(reduction='sum')
    criterion_context_align = LabelSmoothingLoss(reduction='sum', smoothing=0.5)
    criterion_tokens = LabelSmoothingLoss(ignore_index=model.embedding_tgt.word_padding_idx,
                                          reduction='sum', apply_logsoftmax=False)

    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align = [], [], [], [], []
    entry_count, src_max_length, tgt_max_length = 0, 0, 0
    true_batch = []

    if args.verbose == 'True':
        progress_bar = tqdm(train_iter)
    else:
        progress_bar = train_iter
    print('Begin:')
    for epoch in range(args.max_epoch):
        for batch in progress_bar:
            if global_step > args.max_step:
                print('Finish training.')
                break

            model.train()
            raw_src, raw_tgt, _, _, _ = batch

            src_max_length = max(src_max_length, raw_src.shape[0])
            tgt_max_length = max(tgt_max_length, raw_tgt.shape[0])
            entry_count += raw_tgt.shape[1]

            if (src_max_length + tgt_max_length) * entry_count < args.batch_size_token:
                true_batch.append(batch)
            else:
                # Accumulate Batch
                src, tgt, gt_context_alignment, gt_nonreactive_mask, graph_packs = accumulate_batch(true_batch)

                bond, _ = graph_packs
                src, tgt, bond, gt_context_alignment, gt_nonreactive_mask = \
                    src.to(args.device), tgt.to(args.device), bond.to(args.device), \
                    gt_context_alignment.to(args.device), gt_nonreactive_mask.to(args.device)
                del true_batch
                torch.cuda.empty_cache()

                p = np.random.rand()
                if p < anneal_prob(global_step):
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores = model(src, tgt, bond, None)
                else:
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores = model(src, tgt, bond, gt_nonreactive_mask)

                # Loss for language modeling:
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # Loss for atom-level reaction center identification:
                reaction_center_attn = ~gt_nonreactive_mask
                pred_atom_rc_prob = atom_rc_scores.view(-1)
                gt_atom_rc_label = reaction_center_attn.view(-1)

                # Loss for edge-level reaction center identification:
                if bond_rc_scores is not None:
                    pair_indices = torch.where(bond.sum(-1) > 0)
                    pred_bond_rc_prob = bond_rc_scores.view(-1)
                    gt_bond_rc_label = (reaction_center_attn[[pair_indices[1], pair_indices[0]]] & reaction_center_attn[
                        [pair_indices[2], pair_indices[0]]])
                    loss_bond_rc = criterion_bond_rc(pred_bond_rc_prob, gt_bond_rc_label.float())
                else:
                    loss_bond_rc = torch.zeros(1).to(src.device)

                # Loss for context alignment:
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # Compute all loss:
                loss_token = criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = criterion_atom_rc(pred_atom_rc_prob, gt_atom_rc_label.float())
                loss_context_align = 0
                # for context_score in context_scores:
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])

                loss_context_align += criterion_context_align(pred_context_align_logit, gt_context_align_label) 

                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history_all.append(loss.item())
                loss_history_token.append(loss_token.item())
                loss_history_arc.append(loss_atom_rc.item())
                loss_history_brc.append(loss_bond_rc.item())
                loss_history_align.append(loss_context_align.item())

                csv_file_name = f'log_training_file_retroformer_readretro_{args.max_epoch}_epochs'
                
                if getattr(args, 'lowerlr', False):
                    csv_file_name += f'_{args.times}x_lowerLR.csv'
                elif getattr(args, 'extralayer', False):
                    csv_file_name += '_ExtraLayer.csv'
                else:
                    csv_file_name += '.csv'

                if not os.path.exists(csv_file_name):
                    with open(csv_file_name, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(['Epoch', 'Global Step', 'Loss', 'NLL-Loss', 'Arc-Loss', 'Brc-Loss', 'Align-Loss'])


                if global_step % args.report_per_step == 0:
                    print_line = "[Epoch {} Iter {}] Loss {} NLL-Loss {} Rc-Loss {} {} Align-Loss {}".format(
                        epoch, global_step,
                        round(np.mean(loss_history_all), 4), round(np.mean(loss_history_token), 4),
                        round(np.mean(loss_history_arc), 4), round(np.mean(loss_history_brc), 4),
                        round(np.mean(loss_history_align), 4))
                    print(print_line)

                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')
                        
                    with open(csv_file_name, 'a+', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([
                            epoch,
                            global_step,
                            round(np.mean(loss_history_all), 4),
                            round(np.mean(loss_history_token), 4),
                            round(np.mean(loss_history_arc), 4),
                            round(np.mean(loss_history_brc), 4),
                            round(np.mean(loss_history_align), 4)
                        ])
                        
                    loss_history_all, loss_history_token, loss_history_arc, loss_history_brc, loss_history_align = [], [], [], [], []

                if global_step % args.save_per_step == 0:
                    checkpoint_path = args.checkpoint_dir + '/model_{}.pt'.format(global_step)
                    torch.save({'model': model.state_dict(), 'step': global_step, 'optim': optimizer.state_dict()}, checkpoint_path)
                    print('Checkpoint saved to {}'.format(checkpoint_path))

                if global_step % args.val_per_step == 0:
                    accuracy_arc, accuracy_brc, accuracy_token = validate(model, val_iter, model.embedding_tgt.word_padding_idx)
                    print_line = 'Validation accuracy: {} - {} - {}'.format(round(accuracy_arc, 4), round(accuracy_brc, 4), round(accuracy_token, 4))
                    print(print_line)

                    validation_accuracy_overtime = pd.concat([validation_accuracy_overtime, pd.DataFrame({
                        'epoch': [epoch],
                        'global_step': [global_step],
                        'accuracy_arc': [accuracy_arc],
                        'accuracy_brc': [accuracy_brc],
                        'accuracy_token': [accuracy_token]
                    })])
 
                    validation_accuracy_file_name = f'validation_accuracy_overtime_retroformer_readretro_{args.max_epoch}_epochs'
                    
                    if getattr(args, 'lowerlr', False):
                        validation_accuracy_file_name += f'_{args.times}x_lowerLR.csv'
                    elif getattr(args, 'extralayer', False):
                        validation_accuracy_file_name += '_ExtraLayer.csv'
                    else:
                        validation_accuracy_file_name += '.csv'
                    
                    validation_accuracy_overtime.to_csv(validation_accuracy_file_name, index=False)

                    with open(log_file_name, 'a+') as f:
                        f.write(print_line)
                        f.write('\n')

                # Restart Accumulation
                global_step += 1
                true_batch = [batch]
                entry_count, src_max_length, tgt_max_length = raw_src.shape[1], raw_src.shape[0], raw_tgt.shape[0]

if __name__ == '__main__':
    device = sys.argv[1]
    args.device = device
    mode_flag = sys.argv[3].lower()

    allowed_devices = [f'cuda:{i}' for i in range(5)]

    if device in allowed_devices:
        args.device = device
    else:
        raise ValueError(f"Invalid device {device}. Allowed devices are: {', '.join(allowed_devices)}")

    if mode_flag == 'lowerlr':
        times = int(sys.argv[4])
        if times not in [10, 100, 1000]:
            raise ValueError(f"Invalid times value {times}. Allowed values are 10, 100, or 1000.")
        
        args.lowerlr = True
        args.times = times

        epochs_dict = {'cuda:3': 700, 'cuda:4': 1000}

        if device in epochs_dict and device in checkpoint_dict:
            args.max_epoch = epochs_dict[device]
        else:
            raise ValueError(f"Device {device} not supported for lowerLR")

        args.checkpoint_dir = f'../checkpoint_retroformer_readretro_{args.max_epoch}_{times}x_lowerLR'
        args.intermediate_dir = '../intermediate_retroformer_readretro'

        args.data_dir = '../data_plantcyc'
        main(args)

    elif mode_flag == 'extralayer':
        args.extralayer = True
        args.device = device

        epochs_dict = {'cuda:3': 700, 'cuda:4': 1000}

        if device in epochs_dict and device in checkpoint_dict:
            args.max_epoch = epochs_dict[device]
        else:
            raise ValueError(f"Device {device} not supported for ExtraLayer")

        args.checkpoint_dir = f'../checkpoint_retroformer_readretro_{args.max_epoch}_ExtraLayer'
        args.intermediate_dir = '../intermediate_retroformer_readretro'
        args.encoder_num_layers = 9
        args.decoder_num_layers = 9
        
        args.data_dir = '../data_plantcyc'
        main(args)

    elif mode_flag == 'standard':
        args.lowerlr = False
        args.times = None
        args.extralayer = False
        args.device = device

        epochs_dict = {'cuda:0': 100, 'cuda:1': 300, 'cuda:2': 500, 'cuda:3': 700, 'cuda:4': 1000}

        if device in allowed_devices:
            args.device = device
            args.max_epoch = epochs_dict[device]
        else:
            raise ValueError(f"Invalid device {device}. Allowed devices are: {', '.join(allowed_devices)}")

        args.checkpoint_dir = f'../checkpoint_retroformer_readretro_{args.max_epoch}'
        args.intermediate_dir = '../intermediate_retroformer_readretro'
        
        args.data_dir = '../data_plantcyc'
        main(args)

    else:
        raise ValueError(f"Invalid mode flag {mode_flag}.")