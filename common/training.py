"""A small BCE training loop written solely for the public synthetic example."""
import datetime
import json
from pathlib import Path
import random
import torch
from .data import batch_tensors, seed_everything
from .metrics import ranking_metrics


@torch.no_grad()
def evaluate(model, cases, n_items, device, ks, batch_size, mode):
    model.eval()
    chunks = {}
    for begin in range(0, len(cases), batch_size):
        ids, mask, _ = batch_tensors(cases[begin:begin+batch_size], n_items, device)
        for name, values in model(ids, mask).items():
            chunks.setdefault(name, []).append(values.cpu())
    scores = {name: torch.cat(values).numpy() for name, values in chunks.items()}
    if mode == 'R01':
        import numpy as np
        scores['union'] = np.maximum(scores['main'], scores['adjunct'])
    keys = {'union': 'acupoint_ids', 'main': 'main_ids', 'adjunct': 'adjunct_ids'}
    return {name: ranking_metrics(values, [case[keys[name]] for case in cases], ks)
            for name, values in scores.items()}


def display(prefix, metrics, ks):
    for role, values in metrics.items():
        text = ' | '.join(name + ': ' + ', '.join('@%d=%.5f' % (k, values[str(k)][name]) for k in ks)
                          for name in ('recall', 'precision', 'ndcg'))
        print(prefix + ' ' + role + ' ' + text, flush=True)


def run(args, mode, data_class, model_class):
    if mode not in ('gcn', 'R01'):
        raise ValueError('Unknown demo mode')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; pass --device cpu')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.output_dir) / (mode + '_synthetic_demo_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    folds = range(5) if args.all_folds else [args.fold]
    results = []
    for fold in folds:
        seed_everything(args.seed + fold)
        data = data_class(args.data_path, fold)
        if max(args.Ks) > data.n_items:
            raise ValueError('Requested K exceeds number of synthetic candidates')
        model = model_class(data.n_users, data.n_items, args.embed_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        history = []
        for epoch in range(1, args.epoch + 1):
            model.train()
            order = list(range(len(data.train)))
            random.Random(args.seed + fold * 10000 + epoch).shuffle(order)
            total = 0.
            for start in range(0, len(order), args.batch_size):
                batch = [data.train[i] for i in order[start:start+args.batch_size]]
                ids, mask, targets = batch_tensors(batch, data.n_items, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids, mask)
                loss = sum(loss_fn(values, targets[name]) for name, values in logits.items())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite synthetic-demo loss')
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            history.append({'epoch': epoch, 'loss': total / len(data.train)})
            print(f'[synthetic_demo {mode} fold={fold}] epoch={epoch}/{args.epoch} loss={history[-1]["loss"]:.6f}', flush=True)
        train = evaluate(model, data.train, data.n_items, device, args.Ks, args.batch_size, mode)
        test = evaluate(model, data.test, data.n_items, device, args.Ks, args.batch_size, mode)
        display('train', train, args.Ks)
        display('test ', test, args.Ks)
        record = dict(protocol='synthetic_demo_not_paper_reproduction', model=mode, fold=fold,
                      epoch=args.epoch, train_cases=len(data.train), test_cases=len(data.test),
                      train=train, test=test, history=history)
        (output/f'fold_{fold}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
        if args.save_checkpoint:
            torch.save({'protocol': record['protocol'], 'model_state': model.state_dict(),
                        'epoch': args.epoch, 'fold': fold}, output/f'fold_{fold}.pt')
        results.append(record)
    (output/'summary.json').write_text(json.dumps({'protocol':'synthetic_demo_not_paper_reproduction', 'folds':results}, indent=2)+'\n', encoding='utf8')
    print('synthetic_demo_outputs=' + str(output), flush=True)
    return results
