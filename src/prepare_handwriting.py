"""Reproduce manually reviewed crops from the supplied 1550x779 photograph.
Coordinates are chosen from visible digit boundaries, never model predictions.
The original image is preserved; no strokes are generated or redrawn.
"""
import csv
import json
from pathlib import Path
from PIL import Image, ImageOps

BOXES = [
    (190, 350, 315, 490), (324, 350, 452, 475),
    (441, 330, 504, 470), (512, 340, 622, 475),
    (622, 335, 733, 462), (743, 322, 839, 457),
    (840, 340, 941, 464), (944, 337, 1042, 468),
    (1039, 333, 1139, 467), (1145, 336, 1208, 452),
    (1209, 333, 1337, 455),
]
# Human transcription of the photograph; not a model output.
TRANSCRIPTION = '22120606015'


def main():
    root = Path('handwriting')
    source = root/'raw/student_number.jpg'
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.size != (1550,779):
            raise ValueError('These reviewed crop coordinates require the original 1550x779 photograph.')
        (root/'digits').mkdir(parents=True, exist_ok=True)
        for i, box in enumerate(BOXES,1):
            image.crop(box).save(root/f'digits/{i:02d}.png')
    with (root/'labels.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['file','label'])
        for i,label in enumerate(TRANSCRIPTION,1): writer.writerow([f'digits/{i:02d}.png',label])
    (root/'crop_metadata.json').write_text(json.dumps({
        'original':'raw/student_number.jpg', 'original_size':[1550,779],
        'coordinate_order':'left, top, right, bottom; right/bottom exclusive',
        'method':'manually reviewed rectangular crops; no stroke modification',
        'labels_source':'human visual transcription; student confirmation recommended',
        'transcription':TRANSCRIPTION, 'boxes':BOXES,
    },ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
