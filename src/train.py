"""Run controlled experiments, freeze validation selection, then evaluate test once."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.core import MLP, normalize, load_train, split_indices, seed_all, sha256


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    total_loss, correct, predictions = 0.0, 0, []
    for start in range(0, len(y), 1024):
        xb, yb = x[start:start+1024].to(device), y[start:start+1024].to(device)
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction='sum').item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        predictions.extend(pred.cpu().tolist())
    return total_loss / len(y), correct / len(y), np.array(predictions)


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; use a new --output to preserve results.')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    dump(out/'config.json', config)
    x, y = load_train(args.data)
    train_idx, val_idx = split_indices(y, config['split_seed'])
    np.savez_compressed(out/'split_indices.npz', train=train_idx, validation=val_idx)
    xt, xv = torch.from_numpy(normalize(x[train_idx])), torch.from_numpy(normalize(x[val_idx]))
    yt, yv = torch.tensor(y[train_idx], dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long)
    device = torch.device(args.device)
    dump(out/'environment.json', {'python':platform.python_version(), 'torch':str(torch.__version__),
         'numpy':np.__version__, 'platform':platform.platform(), 'device':str(device),
         'threads':args.threads, 'cpu':platform.processor(), 'mnist_sha256':sha256(args.data),
         'train_n':len(yt),'validation_n':len(yv),'test_n':10000,
         'train_class_counts':np.bincount(yt.numpy()).tolist(),
         'validation_class_counts':np.bincount(yv.numpy()).tolist()})
    rows = []
    for exp in config['experiments']:
        for seed in config['seeds']:
            seed_all(seed)
            folder = out / f"{exp['name']}_seed{seed}"
            folder.mkdir()
            model = MLP(exp['hidden'], exp['dropout']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=exp['lr'])
            generator = torch.Generator().manual_seed(seed)
            history, best_loss = [], float('inf')
            if device.type == 'cuda': torch.cuda.synchronize()
            started = time.perf_counter()
            for epoch in range(1, config['epochs']+1):
                model.train()
                order = torch.randperm(len(yt), generator=generator)
                for idx in order.split(config['batch_size']):
                    xb, yb = xt[idx].to(device), yt[idx].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                # Same checkpoint and eval mode for comparable training/validation curves.
                tl, ta, _ = evaluate(model, xt, yt, device)
                vl, va, _ = evaluate(model, xv, yv, device)
                history.append(dict(epoch=epoch, train_loss=tl, train_accuracy=ta,
                                    validation_loss=vl, validation_accuracy=va))
                if vl < best_loss:
                    best_loss, best_epoch = vl, epoch
                    torch.save({'state_dict':{k:v.cpu().clone() for k,v in model.state_dict().items()},
                                'experiment':exp, 'seed':seed, 'epoch':epoch,
                                'normalization':'uint8 / 255.0'}, folder/'best.pt')
                print(f"{exp['name']} seed={seed} epoch={epoch:02d}: train={ta:.4f}, val={va:.4f}, val_loss={vl:.4f}", flush=True)
            if device.type == 'cuda': torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            dump(folder/'history.json', history)
            fig, axes = plt.subplots(1,2,figsize=(10,3.8))
            for ax, metric in zip(axes, ['loss','accuracy']):
                for prefix, label in [('train','Train'),('validation','Validation')]:
                    ax.plot([h['epoch'] for h in history], [h[f'{prefix}_{metric}'] for h in history],label=label)
                ax.axvline(best_epoch,ls=':',color='gray',label='Best val loss')
                ax.set(xlabel='Epoch',ylabel=metric.title()); ax.legend(); ax.grid(alpha=.2)
            fig.suptitle(f"{exp['name']} / seed {seed}"); fig.tight_layout(); fig.savefig(folder/'curves.png',dpi=160); plt.close(fig)
            best = history[best_epoch-1]
            rows.append({'experiment':exp['name'],'seed':seed,'best_epoch':best_epoch,
                         'parameters':sum(p.numel() for p in model.parameters()),
                         'training_seconds':seconds,**{k:v for k,v in best.items() if k!='epoch'}})
    # Freeze every decision BEFORE accessing test arrays.
    means = {exp['name']: float(np.mean([r['validation_loss'] for r in rows if r['experiment']==exp['name']])) for exp in config['experiments']}
    selected = min(means, key=means.get)
    representative = min([r for r in rows if r['experiment']==selected], key=lambda r:r['validation_loss'])
    selection = {'experiment':selected,'seed':representative['seed'],
                 'checkpoint':f"{selected}_seed{representative['seed']}/best.pt",
                 'mean_validation_losses':means,'selection_uses_test':False}
    dump(out/'selection.json', selection)
    dump(out/'validation_results.json', rows)
    with np.load(args.data, allow_pickle=False) as data:
        test_images, test_labels = data['x_test'], data['y_test']
    assert test_images.shape == (10000,28,28) and test_labels.shape == (10000,)
    xx, yy = torch.from_numpy(normalize(test_images)), torch.tensor(test_labels,dtype=torch.long)
    for row in rows:
        folder = out/f"{row['experiment']}_seed{row['seed']}"
        checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        exp = checkpoint['experiment']
        model = MLP(exp['hidden'],exp['dropout']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        loss, acc, pred = evaluate(model, xx, yy, device)
        row.update(test_loss=loss,test_accuracy=acc)
        np.save(folder/'test_predictions.npy',pred)
        if row['experiment']==selected and row['seed']==representative['seed']:
            dump(out/'classification_report.json',classification_report(test_labels,pred,output_dict=True,zero_division=0))
            fig,ax=plt.subplots(figsize=(7,6)); ConfusionMatrixDisplay(confusion_matrix(test_labels,pred)).plot(ax=ax,cmap='Blues',colorbar=False)
            ax.set_title('Selected model — MNIST test'); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
            errors=np.flatnonzero(pred!=test_labels)[:24]
            fig,axes=plt.subplots(3,8,figsize=(12,5))
            for ax in axes.flat: ax.axis('off')
            for ax,i in zip(axes.flat,errors):
                ax.imshow(test_images[i],cmap='gray'); ax.set_title(f'True {test_labels[i]} / Pred {pred[i]}',fontsize=9)
            fig.tight_layout(); fig.savefig(out/'test_errors.png',dpi=160); plt.close(fig)
    dump(out/'results.json',rows)
    with open(out/'results.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Finished. Selected on validation:',selection,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiments.json')
    p.add_argument('--data',default='data/mnist.npz')
    p.add_argument('--output',default='results/run1')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
