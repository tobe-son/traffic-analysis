import unittest
import numpy as np
from research.audio_protocol import prepare, grouped_folds, fit_scale

class AudioProtocolTests(unittest.TestCase):
    def test_stereo_retains_relative_level(self):
        t = np.arange(48000) / 48000
        s = np.sin(2*np.pi*1000*t)
        y = prepare(np.column_stack([s, .25*s]), 48000, 8000)
        self.assertEqual(y.shape, (8000, 2))
        np.testing.assert_allclose(y[:, 1], .25*y[:, 0], atol=1e-12)
    def test_alias_rejected(self):
        t = np.arange(48000)/48000
        s = np.sin(2*np.pi*7000*t)
        y = prepare(s, 48000, 8000, 'left')[100:-100]
        self.assertLess(np.sqrt(np.mean(y*y)), .01)
    def test_mono_cannot_be_called_stereo(self):
        with self.assertRaises(ValueError):
            prepare(np.ones(100), 16000, 8000)
    def test_groups_disjoint(self):
        folds = grouped_folds([{'recording_id':str(i), 'vehicle':str(i//2)} for i in range(6)], 'vehicle')
        for fold in folds:
            self.assertFalse(set(fold['train']) & set(fold['test']))
            self.assertEqual(len(fold['test']), 2)
    def test_train_scaling(self):
        self.assertEqual(fit_scale([0, 2]), (0, 2))
    def test_missing_group_rejected(self):
        with self.assertRaises(ValueError):
            grouped_folds([{'recording_id':'a'}], 'vehicle')
