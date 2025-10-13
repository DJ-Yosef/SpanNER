import torch

def span_f1(predicts, span_label_ltoken, real_span_mask_ltoken):
    pred_label_idx = torch.max(predicts, dim=-1)[1]  # (bs, n_span)
    pred_label_mask = (pred_label_idx != 0)
    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct * pred_label_mask * real_span_mask_ltoken.bool()

    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx != 0)
    total_golden = torch.sum(span_label_ltoken != 0)

    return torch.stack([correct_pred, total_pred, total_golden])


def span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken):
    pred_label_idx = torch.max(predicts, dim=-1)[1]
    span_probs = predicts.tolist()

    nonO_idxs2labs, nonO_kidxs_all, pred_label_idx_new = get_pruning_predIdxs(
        pred_label_idx, all_span_idxs, span_probs
    )

    pred_label_idx = pred_label_idx_new.to(predicts.device)
    pred_label_mask = (pred_label_idx != 0)
    all_correct = pred_label_idx == span_label_ltoken
    all_correct = all_correct * pred_label_mask * real_span_mask_ltoken.bool()

    correct_pred = torch.sum(all_correct)
    total_pred = torch.sum(pred_label_idx != 0)
    total_golden = torch.sum(span_label_ltoken != 0)

    return torch.stack([correct_pred, total_pred, total_golden]), pred_label_idx


def get_predict(args, all_span_word, words, predicts, span_label_ltoken, all_span_idxs):
    pred_label_idx = torch.max(predicts, dim=-1)[1]
    idx2label = {int(v): k for k, v in args.label2idx.items()}

    batch_preds = []
    for span_idxs, word, ws, lps, lts in zip(all_span_idxs, words, all_span_word, pred_label_idx, span_label_ltoken):
        text = ' '.join(word) + "\t"
        for sid, w, lp, lt in zip(span_idxs, ws, lps, lts):
            if lp != 0 or lt != 0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w) + f':: {int(sidx)},{int(eidx+1)}:: {tlabel}:: {plabel}\t'
                text += ctext
        batch_preds.append(text)
    return batch_preds


def get_predict_prune(args, all_span_word, words, predicts_new, span_label_ltoken, all_span_idxs):
    idx2label = {int(v): k for k, v in args.label2idx.items()}
    batch_preds = []
    for span_idxs, word, ws, lps, lts in zip(all_span_idxs, words, all_span_word, predicts_new, span_label_ltoken):
        text = ' '.join(word) + "\t"
        for sid, w, lp, lt in zip(span_idxs, ws, lps, lts):
            if lp != 0 or lt != 0:
                plabel = idx2label[int(lp.item())]
                tlabel = idx2label[int(lt.item())]
                sidx, eidx = sid
                ctext = ' '.join(w) + f':: {int(sidx)},{int(eidx+1)}:: {tlabel}:: {plabel}\t'
                text += ctext
        batch_preds.append(text)
    return batch_preds


def has_overlapping(idx1, idx2):
    return not (idx1[0] > idx2[1] or idx2[0] > idx1[1])


def clean_overlapping_span(idxs_list, nonO_idxs2prob):
    kidxs = []
    didxs = []
    for i in range(len(idxs_list) - 1):
        idx1 = idxs_list[i]
        keep = True
        for j in range(i + 1, len(idxs_list)):
            idx2 = idxs_list[j]
            if has_overlapping(idx1, idx2):
                prob1 = nonO_idxs2prob[idx1]
                prob2 = nonO_idxs2prob[idx2]
                if prob1 < prob2:
                    keep = False
                    didxs.append(idx1)
                elif prob1 == prob2:
                    len1 = idx1[1] - idx1[0] + 1
                    len2 = idx2[1] - idx2[0] + 1
                    if len1 < len2:
                        keep = False
                        didxs.append(idx1)
        if keep:
            conflict = False
            for idx in kidxs:
                if has_overlapping(idx1, idx):
                    conflict = True
                    if nonO_idxs2prob[idx1] > nonO_idxs2prob[idx]:
                        kidxs.remove(idx)
                        kidxs.append(idx1)
                    break
            if not conflict:
                kidxs.append(idx1)
    if len(didxs) == 0 or idxs_list[-1] not in didxs:
        kidxs.append(idxs_list[-1])
    return kidxs


def get_pruning_predIdxs(pred_label_idx, all_span_idxs, span_probs):
    nonO_kidxs_all = []
    nonO_idxs2labs = []
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        nonO_idxs2lab = {}
        nonO_idxs2prob = {}
        nonO_idxs = []
        for j, (plb, idx) in enumerate(zip(bs, idxs)):
            plb = int(plb.item())
            if plb != 0:
                nonO_idxs2lab[idx] = plb
                nonO_idxs2prob[idx] = span_probs[i][j][plb]
                nonO_idxs.append(idx)
        nonO_idxs2labs.append(nonO_idxs2lab)
        nonO_kidxs = clean_overlapping_span(nonO_idxs, nonO_idxs2prob) if nonO_idxs else []
        nonO_kidxs_all.append(nonO_kidxs)

    pred_label_idx_new = []
    n_span = pred_label_idx.size(1)
    for i, (bs, idxs) in enumerate(zip(pred_label_idx, all_span_idxs)):
        pred_row = [
            int(bs[j].item()) if idx in nonO_kidxs_all[i] else 0
            for j, idx in enumerate(idxs)
        ]
        while len(pred_row) < n_span:
            pred_row.append(0)
        pred_label_idx_new.append(pred_row)

    return nonO_idxs2labs, nonO_kidxs_all, torch.LongTensor(pred_label_idx_new)
