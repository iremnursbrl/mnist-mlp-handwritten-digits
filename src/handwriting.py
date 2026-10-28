"""Predict a student's own digits, provided as separate crops in a manifest."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.core import MLP, normalize


def preprocess(path):
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('L')
    a = np.asarray(im, dtype=np.float32)
    if min(a.shape) < 3:
        raise ValueError('Image is too small.')
    border = np.concatenate([a[0],a[-1],a[:,0],a[:,-1]])
    if np.median(border) > 127:
        a = 255-a
    # Suppress background using robust intensity range, retaining antialiasing.
    background = float(np.percentile(a, 10))
    a = np.maximum(a-background,0)
    peak = float(a.max())
    if peak < 15:
        raise ValueError(f'Blank or low-contrast image: {path}')
    a = np.clip(a / peak * 255,0,255)
    mask = a > max(30, float(np.percentile(a,99))*.2)
    yy,xx = np.nonzero(mask)
    if len(xx) < 3:
        raise ValueError(f'No digit found: {path}')
    a = a[yy.min():yy.max()+1,xx.min():xx.max()+1]
    height,width=a.shape
    scale=20/max(height,width)
    size=(max(1,round(width*scale)),max(1,round(height*scale)))
    a=np.asarray(Image.fromarray(a.astype(np.uint8)).resize(size,Image.Resampling.LANCZOS),dtype=np.float32)
    canvas=np.zeros((28,28),dtype=np.float32)
    h,w=a.shape; top,left=(28-h)//2,(28-w)//2
    canvas[top:top+h,left:left+w]=a
    cy,cx=center_of_mass(canvas)
    canvas=shift(canvas,(13.5-cy,13.5-cx),order=1,mode='constant',cval=0,prefilter=False)
    return np.clip(np.rint(canvas),0,255).astype(np.uint8)


def run(args):
    manifest=Path(args.manifest)
    with manifest.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest is empty; add your own handwriting first.')
    labels=[]
    for r in rows:
        if r['label'] not in list('0123456789'):
            raise ValueError('Each label must be one digit, 0–9.')
        labels.append(int(r['label']))
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new, empty output folder.')
    out.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=ckpt['experiment']; model=MLP(cfg['hidden'],cfg['dropout'])
    model.load_state_dict(ckpt['state_dict']); model.eval()
    images=np.stack([preprocess(manifest.parent/r['file']) for r in rows])
    with torch.no_grad():
        probabilities=model(torch.from_numpy(normalize(images))).softmax(1).numpy()
    predictions=probabilities.argmax(1)
    for i,im in enumerate(images): Image.fromarray(im).save(out/f'digit_{i:02d}_28x28.png')
    result={'status':'evaluated_real_user_supplied_images', 'checkpoint':str(args.checkpoint),
            'true_number':''.join(map(str,labels)), 'predicted_number':''.join(map(str,predictions)),
            'digit_accuracy':float(np.mean(predictions==labels)),
            'exact_number_match':bool(np.array_equal(predictions,labels)),
            'n_digits':len(labels), 'digits':[]}
    for r,pred,probs in zip(rows,predictions,probabilities):
        result['digits'].append({'file':r['file'],'label':int(r['label']),'prediction':int(pred),
                                'softmax_score':float(probs[pred]),'probabilities':probs.tolist()})
    (out/'predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=min(8,len(rows)); count=(len(rows)+cols-1)//cols
    fig,axes=plt.subplots(count,cols,figsize=(2*cols,2.5*count),squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax,im,label,pred in zip(axes.flat,images,labels,predictions):
        ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        ax.set_title(f'True: {label}\nPred: {pred}',color='green' if label==pred else 'crimson')
    fig.tight_layout(); fig.savefig(out/'predictions.png',dpi=160); plt.close(fig)
    report=['# Kendi el yazısı değerlendirmesi','',f"Gerçek numara: `{result['true_number']}`",
            f"Tahmin: `{result['predicted_number']}`",f"Rakam doğruluğu: %{100*result['digit_accuracy']:.2f}",
            f"Numaranın tamamı doğru mu: {result['exact_number_match']}",'',
            '![Tahminler](predictions.png)','',
            'Softmax skorları kalibre edilmiş güven olasılığı değildir. Tek numaradaki az sayıdaki rakam, genel el yazısı performansını ölçmek için yeterli değildir.',
            'Bu görüntüler eğitimde veya model seçiminde kullanılmamıştır. Ön işleme sonucu 28×28 görseller tek tek incelenmelidir.']
    (out/'handwriting_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='handwriting/labels.csv')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='results/handwriting')
    run(p.parse_args())
