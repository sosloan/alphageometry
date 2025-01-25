# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for lm_inference.py."""
import os
import unittest

from absl import flags
from absl.testing import absltest
import lm_inference as lm


_DATA_PATH = flags.DEFINE_string('data_path', '', 'path to ckpt and vocab.')
_MELIAD_PATH = flags.DEFINE_string(
    'meliad_path', '', 'path to meliad repository.'
)  # pylint: disable=line-too-long


class LmInferenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    gin_file = [
        'base_htrans.gin',
        'size/medium_150M.gin',
        'options/positions_t5.gin',
        'options/lr_cosine_decay.gin',
        'options/seq_1024_nocache.gin',
        'geometry_150M_generate.gin',
    ]

    gin_param = [
        'DecoderOnlyLanguageModelGenerate.output_token_losses=True',
        'TransformerTaskConfig.batch_size=2',
        'TransformerTaskConfig.sequence_length=128',
        'Trainer.restore_state_variables=False',
    ]

    gin_search_paths = [
        os.path.join(_MELIAD_PATH.value, 'transformer/configs'),
        os.getcwd(),
    ]

    vocab_path = os.path.join(_DATA_PATH.value, 'geometry.757.model')

    lm.parse_gin_configuration(gin_file, gin_param, gin_paths=gin_search_paths)

    cls.loaded_lm = lm.LanguageModelInference(
        vocab_path, _DATA_PATH.value, mode='beam_search'
    )

  def test_lm_decode(self):
    outputs = LmInferenceTest.loaded_lm.beam_decode(
        '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c'
        ' {F1} x00',
        eos_tokens=[';'],
    )
    self.assertEqual(
        outputs['seqs_str'],
        ['e : D a b c e 02 D a c b e 03 ;', 'e : C a c e 02 C b d e 03 ;'],
    )

  def test_lm_score_may_fail_numerically_for_external_meliad(self):
    outputs = LmInferenceTest.loaded_lm.beam_decode(
        '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c'
        ' {F1} x00',
        eos_tokens=[';'],
    )
    self.assertEqual(
        outputs['scores'],
        [-1.18607294559478759765625, -1.10228693485260009765625],
    )

  def test_lm_encode(self):
    inputs_str = "This is a test."
    encoded = LmInferenceTest.loaded_lm.encode(inputs_str)
    self.assertIsInstance(encoded, list)
    self.assertTrue(all(isinstance(i, int) for i in encoded))

  def test_lm_encode_list(self):
    inputs_strs = ["This is a test.", "Another test."]
    encoded_list = LmInferenceTest.loaded_lm.encode_list(inputs_strs)
    self.assertIsInstance(encoded_list, list)
    self.assertTrue(all(isinstance(i, int) for i in encoded_list))

  def test_lm_decode_list(self):
    tokens = [1, 2, 3, 4]
    decoded_list = LmInferenceTest.loaded_lm.decode_list(tokens)
    self.assertIsInstance(decoded_list, list)
    self.assertTrue(all(isinstance(s, str) for s in decoded_list))

  def test_lm_call(self):
    inputs = LmInferenceTest.loaded_lm.encode("{S} a : ; b : ; c : ; d :")
    inputs = jax.numpy.array([inputs] * LmInferenceTest.loaded_lm.batch_size)
    metrics = LmInferenceTest.loaded_lm.call(inputs)
    self.assertIsInstance(metrics, dict)
    self.assertIn('finished_seqs', metrics)
    self.assertIn('finished_scores', metrics)

  def test_lm_beam_decode_with_dstate(self):
    inputs = "{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c"
    dstate = None
    outputs = LmInferenceTest.loaded_lm.beam_decode(inputs, dstate=dstate)
    self.assertIsInstance(outputs, dict)
    self.assertIn('finished_seqs', outputs)
    self.assertIn('finished_scores', outputs)
    self.assertIn('dstate', outputs)


if __name__ == '__main__':
  absltest.main()
