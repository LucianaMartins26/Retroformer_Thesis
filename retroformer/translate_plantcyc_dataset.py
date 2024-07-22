from retroformer.utils.smiles_utils import *
from retroformer.utils.translate_utils import translate_batch_original, translate_batch_stepwise
from retroformer.utils.build_utils import build_model, build_iterator, load_checkpoint

import re
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from tqdm import tqdm
import argparse

torch.cuda.empty_cache()

class Args:
    def __init__(self):
        self.device = 'cuda'
        self.batch_size_trn = 2
        self.batch_size_val = 2
        self.data_dir = '../data_plantcyc'
        self.intermediate_dir = '../intermediate'
        self.checkpoint_dir = '../checkpoint_untyped'
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
        self.beam_size = 5
        self.use_template = 'False'
        self.stepwise = 'False'
        self.max_epoch = 30

args = Args()

def validate(model, val_iter, pad_idx=1):
    pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
    pred_arc_list, gt_arc_list = [], []
    pred_brc_list, gt_brc_list = [], []
    model.eval()
    for batch in tqdm(val_iter):
        src, tgt, gt_context_alignment, gt_nonreactive_mask, graph_packs = batch
        bond, _ = graph_packs

        # Infer:
        with torch.no_grad():
            scores, atom_rc_scores, bond_rc_scores, context_alignment = \
                model(src, tgt, bond)
            context_alignment = F.softmax(context_alignment[-1], dim=-1)

        # Atom-level reaction center accuracy:
        pred_arc = (atom_rc_scores.squeeze(2) > 0.5).bool()
        pred_arc_list += list(~pred_arc.view(-1).cpu().numpy())
        gt_arc_list += list(gt_nonreactive_mask.view(-1).cpu().numpy())

        # Bond-level reaction center accuracy:
        if bond_rc_scores is not None:
            pred_brc = (bond_rc_scores > 0.5).bool()
            pred_brc_list += list(pred_brc.view(-1).cpu().numpy())

        pair_indices = torch.where(bond.sum(-1) > 0)
        rc = ~gt_nonreactive_mask
        gt_bond_rc_label = (rc[[pair_indices[1], pair_indices[0]]] & rc[[pair_indices[2], pair_indices[0]]])
        gt_brc_list += list(gt_bond_rc_label.view(-1).cpu().numpy())

        # Token accuracy:
        pred_token_logit = scores.view(-1, scores.size(2))
        _, pred_token_label = pred_token_logit.topk(1, dim=-1)
        gt_token_label = tgt[1:].view(-1)
        pred_token_list.append(pred_token_label[gt_token_label != pad_idx])
        gt_token_list.append(gt_token_label[gt_token_label != pad_idx])

    pred_tokens = torch.cat(pred_token_list).view(-1)
    gt_tokens = torch.cat(gt_token_list).view(-1)

    if bond_rc_scores is not None:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               np.mean(np.array(pred_brc_list) == np.array(gt_brc_list)), \
               (pred_tokens == gt_tokens).float().mean().item()
    else:
        return np.mean(np.array(pred_arc_list) == np.array(gt_arc_list)), \
               0, \
               (pred_tokens == gt_tokens).float().mean().item()
    
def _test_string_accuracies(args, model_name):
    
    train_iter, val_iter, vocab_itos_src, vocab_itos_tgt = build_iterator(args, train=True, sample=False, augment=True)
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    _, _, model = load_checkpoint(args, model)
    train_accuracy_arc, train_accuracy_brc, train_accuracy_token = validate(model, train_iter, model.embedding_tgt.word_padding_idx)
    val_accuracy_arc, val_accuracy_brc, val_accuracy_token = validate(model, val_iter, model.embedding_tgt.word_padding_idx)
    test_iter, dataset = build_iterator(args, train=False, sample=False)
    test_accuracy_arc, test_accuracy_brc, test_accuracy_token = validate(model, test_iter, model.embedding_tgt.word_padding_idx)

    model_results = pd.DataFrame({'Model': [model_name], 'Train_Accuracy_Arc': [train_accuracy_arc], 
                                            'Train_Accuracy_Brc': [train_accuracy_brc], 'Train_Accuracy_Token': [train_accuracy_token],
                                            'Val_Accuracy_Arc': [val_accuracy_arc], 'Val_Accuracy_Brc': [val_accuracy_brc],
                                            'Val_Accuracy_Token': [val_accuracy_token], 'Test_Accuracy_Arc': [test_accuracy_arc],
                                            'Test_Accuracy_Brc': [test_accuracy_brc], 'Test_Accuracy_Token': [test_accuracy_token]})

    return model_results

    
def translate(iterator, model, dataset):
    ground_truths = []
    generations = []
    invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
    invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']]
    # Translate:
    for batch in tqdm(iterator, total=len(iterator)):
        src, tgt, _, _, _ = batch

        if args.stepwise == 'False':
            # Original Main:
            pred_tokens, pred_scores = translate_batch_original(model, batch, beam_size=args.beam_size, invalid_token_indices=invalid_token_indices)
            for idx in range(batch[0].shape[1]):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
                hypos = np.array([''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
                hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
                new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_score)[::-1]

                ground_truths.append(gt)
                generations.append(hypos[ordering])
        else:
            # Stepwise Main:
            # untyped: T=10; beta=0.5, percent_aa=40, percent_ab=40
            # typed: T=10; beta=0.5, percent_aa=40, percent_ab=55
            if args.known_class == 'True':
                percent_ab = 55
            else:
                percent_ab = 40
            pred_tokens, pred_scores, predicts = \
                translate_batch_stepwise(model, batch, beam_size=args.beam_size,
                                         invalid_token_indices=invalid_token_indices,
                                         T=10, alpha_atom=-1, alpha_bond=-1,
                                         beta=0.5, percent_aa=40, percent_ab=percent_ab, k=3,
                                         use_template=args.use_template == 'True',
                                         factor_func=dataset.factor_func,
                                         reconstruct_func=dataset.reconstruct_smi,
                                         rc_path=args.intermediate_dir + '/rt2reaction_center.pk')

            original_beam_size = pred_tokens.shape[1]
            current_i = 0
            for batch_i, predict in enumerate(predicts):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, batch_i], src=False))
                remain = original_beam_size
                beam_size = math.ceil(original_beam_size / len(predict))

                # normalized_reaction_center_score = np.array([pred[1] for pred in predict]) / 10
                hypo_i, hypo_scores_i = [], []
                for j, (rc, rc_score) in enumerate(predict):
                    # rc_score = normalized_reaction_center_score[j]

                    pred_token = pred_tokens[current_i + j]

                    sub_hypo_candidates, sub_score_candidates = [], []
                    for k in range(pred_token.shape[0]):
                        hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                        hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                        hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                        if hypo_smiles_k not in hypo_i:  # only select unique entries
                            sub_hypo_candidates.append(hypo_smiles_k)
                            sub_score_candidates.append(hypo_scores_k)

                    ordering = np.argsort(sub_score_candidates)[::-1]
                    sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                    sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                    hypo_i += sub_hypo_candidates
                    hypo_scores_i += sub_score_candidates

                    remain -= beam_size

                current_i += len(predict)
                ordering = np.argsort(hypo_scores_i)[::-1][:args.beam_size]
                ground_truths.append(gt)
                generations.append(np.array(hypo_i)[ordering])

    return ground_truths, generations

def main(args):
    # Build Data Iterator:
    iterator, dataset = build_iterator(args, train=False)

    # Load Checkpoint Model:
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)

    # Get Output Path:
    dec_version = 'stepwise' if args.stepwise == 'True' else 'vanilla'
    exp_version = 'typed' if args.known_class == 'True' else 'untyped'
    aug_version = '_augment' if 'augment' in args.checkpoint_dir else ''
    tpl_version = '_template' if args.use_template == 'True' else ''
    file_name = '../result/{}_bs_top{}_generation_{}{}{}_{}_epochs_retroformer_test.pk'.format(dec_version, args.beam_size, exp_version,
                                                                    aug_version, tpl_version, args.max_epoch)
    output_path = os.path.join(args.intermediate_dir, file_name)
    print('Output path: {}'.format(output_path))

    # Begin Translating:
    ground_truths, generations = translate(iterator, model, dataset)
    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size))
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1

    with open(output_path, 'wb') as f:
        pickle.dump((ground_truths, generations), f)

    # with open('../result/output_{}_epochs.txt'.format(args.max_epoch), 'w') as f:
    #     for j in range(args.beam_size):
    #         f.write('\n')
    #         f.write('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))
    #     return

    with open('../result/output_{}_epochs_retroformer_test_set.txt'.format(args.max_epoch), 'w') as f:
        for j in range(args.beam_size):
            f.write('\n')
            f.write('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))
        return
    
def test_strings_accuracies():
    class Args:
        def __init__(self):
            self.device = 'cuda'
            self.batch_size_trn = 2
            self.batch_size_val = 2
            self.batch_size_token = 4096
            self.data_dir = '../data_plantcyc'
            self.intermediate_dir = '../intermediate'
            self.checkpoint_dir = '../checkpoint_untyped_cuda1_300'
            self.checkpoint = "model_75000.pt"
            self.encoder_num_layers = 8
            self.decoder_num_layers = 8
            self.d_model = 256
            self.heads = 8
            self.d_ff = 2048
            self.dropout = 0.3
            self.known_class = 'False'
            self.shared_vocab = 'True'
            self.shared_encoder = 'False'
            # self.max_epoch = 30
            self.max_epoch = 1000
            self.max_step = 300000
            self.report_per_step = 200
            self.save_per_step = 2500
            self.val_per_step = 2500
            self.verbose = 'False'

    args = Args()

    dataframe = _test_string_accuracies(args, "300 epochs")
    dataframe.to_csv('../result/300_epochs_accuracies.csv')

def test():
    output_path = '../result/vanilla_bs_top5_generation_untyped_700_epochs.pk'
    with open(output_path, 'rb') as f:
        ground_truths, generations = pickle.load(f)

    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size))
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i]]
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1

    print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))


if __name__ == "__main__":

    predictions = sys.argv[1]
    
    if predictions == 'True':
        device = 'cuda:1'
        args.device = device

        epochs_dict = {'cuda:0' : 100, 'cuda:1' : 300, 'cuda:2' : 500, 'cuda:3' : 700}
        checkpoint_dict = {'cuda:0' : 'model_25000.pt', 'cuda:1': 'model_75000.pt', 'cuda:2': 'model_125000.pt', 'cuda:3' : 'model_175000.pt'}

        args.max_epoch = epochs_dict[device]
        # args.checkpoint = checkpoint_dict[device]

        # if args.known_class == 'True':
        #     args.checkpoint_dir = args.checkpoint_dir + '_typed'
        # else:
        #     args.checkpoint_dir = args.checkpoint_dir + '_untyped'
        # if args.use_template == 'True':
        #     args.stepwise = 'True'

        args.checkpoint_dir = args.checkpoint_dir + '_{}_{}'.format(args.device.replace(':', ''), args.max_epoch)

        main(args)
    else:
        # test_strings_accuracies()
        test()