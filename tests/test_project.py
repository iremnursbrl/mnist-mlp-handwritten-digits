"""Integrity tests use synthetic shapes, not student handwriting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from src.core import MLP, normalize, split_indices
from src.handwriting import preprocess

class ProjectTests(unittest.TestCase):
    def test_split_disjoint_and_complete(self):
        labels=np.tile(np.arange(10),6000)
        train,val=split_indices(labels)
        self.assertEqual(len(train),50000); self.assertEqual(len(val),10000)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(len(set(train)|set(val)),60000)
        np.testing.assert_array_equal(train,split_indices(labels)[0])
        np.testing.assert_array_equal(np.bincount(labels[val]),np.full(10,1000))

    def test_network_logits_and_gradient(self):
        model=MLP([128,64])
        self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in model.modules()),3)
        self.assertFalse(any(isinstance(m,torch.nn.Conv2d) for m in model.modules()))
        x=torch.rand(8,28,28); logits=model(x)
        self.assertEqual(tuple(logits.shape),(8,10))
        loss=torch.nn.functional.cross_entropy(logits,torch.arange(8))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_preprocessing_polarity_and_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'shape.png'
            im=Image.new('L',(90,120),255); d=ImageDraw.Draw(im); d.line([(25,20),(40,100)],fill=0,width=9)
            im.save(p); a=preprocess(p)
            Image.fromarray(255-np.array(im)).save(p); b=preprocess(p)
            self.assertEqual(a.shape,(28,28)); np.testing.assert_array_equal(a,b)
            self.assertEqual(normalize(a).dtype,np.float32)
            self.assertGreater(a.max(),200)
            Image.new('L',(90,120),255).save(p)
            with self.assertRaises(ValueError): preprocess(p)

if __name__=='__main__': unittest.main()
