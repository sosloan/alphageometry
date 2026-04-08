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

"""Unit tests for alphageometry.py."""

import unittest

from absl.testing import absltest
import alphageometry


class AlphaGeometryTest(unittest.TestCase):

  def test_translate_constrained_to_constructive(self):
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('addb')
        ),
        ('on_dia', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('adbc')
        ),
        ('on_tline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'P', list('bcda')
        ),
        ('on_pline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcd')
        ),
        ('on_bline', ['d', 'c', 'b']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcb')
        ),
        ('on_circle', ['d', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bacd')
        ),
        ('eqdistance', ['d', 'c', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'O', list('abcd')
        ),
        ('on_circum', ['d', 'a', 'b', 'c']),
    )

  def test_insert_aux_to_premise(self):
    pstring = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b ? perp a d b c'  # pylint: disable=line-too-long
    auxstring = 'e = on_line e a c, on_line e b d'

    target = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'  # pylint: disable=line-too-long
    self.assertEqual(
        alphageometry.insert_aux_to_premise(pstring, auxstring), target
    )

  def test_beam_queue(self):
    beam_queue = alphageometry.BeamQueue(max_size=2)

    beam_queue.add('a', 1)
    beam_queue.add('b', 2)
    beam_queue.add('c', 3)

    beam_queue = list(beam_queue)
    self.assertEqual(beam_queue, [(3, 'c'), (2, 'b')])

  def test_natural_language_statement(self):
    logical_statement = pr.Dependency(name='cong', args=['A', 'B', 'C', 'D'])
    result = alphageometry.natural_language_statement(logical_statement)
    self.assertEqual(result, 'A is congruent to B, C, and D')

  def test_proof_step_string(self):
    proof_step = pr.Dependency(
        name='cong', args=['A', 'B', 'C', 'D'], premises=[], conclusion=['E']
    )
    refs = {('A', 'B', 'C', 'D'): 1}
    result = alphageometry.proof_step_string(proof_step, refs, last_step=False)
    self.assertEqual(result, 'A is congruent to B, C, and D [01] ⇒ E')

  def test_write_solution(self):
    g = gh.Graph()
    p = pr.Problem(goal=pr.Dependency(name='cong', args=['A', 'B', 'C', 'D']))
    out_file = 'test_output.txt'
    alphageometry.write_solution(g, p, out_file)
    with open(out_file, 'r') as f:
      content = f.read()
    self.assertIn('Solution written to', content)

  def test_get_lm(self):
    ckpt_init = 'path/to/checkpoint'
    vocab_path = 'path/to/vocab'
    result = alphageometry.get_lm(ckpt_init, vocab_path)
    self.assertIsInstance(result, lm.LanguageModelInference)

  def test_run_ddar(self):
    g = gh.Graph()
    p = pr.Problem(goal=pr.Dependency(name='cong', args=['A', 'B', 'C', 'D']))
    out_file = 'test_output.txt'
    result = alphageometry.run_ddar(g, p, out_file)
    self.assertFalse(result)

  def test_try_translate_constrained_to_construct(self):
    string = 'd = perp a b c d ;'
    g = gh.Graph()
    result = alphageometry.try_translate_constrained_to_construct(string, g)
    self.assertEqual(result, 'd = on_tline a b c d')

  def test_check_valid_args(self):
    self.assertTrue(alphageometry.check_valid_args('perp', ['a', 'b', 'c', 'd']))
    self.assertFalse(alphageometry.check_valid_args('perp', ['a', 'b', 'c']))

  def test_run_alphageometry(self):
    model = lm.LanguageModelInference(vocab_path='path/to/vocab', ckpt_init='path/to/checkpoint')
    p = pr.Problem(goal=pr.Dependency(name='cong', args=['A', 'B', 'C', 'D']))
    search_depth = 1
    beam_size = 1
    out_file = 'test_output.txt'
    result = alphageometry.run_alphageometry(model, p, search_depth, beam_size, out_file)
    self.assertFalse(result)

  def test_main(self):
    alphageometry.main(None)

  def test_get_pos_candidates_branch(self):
    website = "http://example.com"
    task = "Test Task"
    curriculum = "Test Curriculum"
    actionable_elements = [{"id": "test_id", "field_name": "test_field"}]
    step = "Test Step"
    model = "gpt-4"
    CoT = False
    result = alphageometry.get_pos_candidates_branch(website, task, curriculum, actionable_elements, step, model, CoT)
    self.assertIn('response', result)

  def test_tree_thought_generation(self):
    website = "http://example.com"
    task = "Test Task"
    curriculum = "Test Curriculum"
    actionable_elements = [{"id": "test_id", "field_name": "test_field"}]
    step = "Test Step"
    model = "gpt-4"
    CoT = False
    branches = 3
    result = alphageometry.tree_thought_generation(website, task, curriculum, actionable_elements, CoT, branches, step, model)
    self.assertIn('response', result)

  def test_semiose_vetter(self):
    actionable_elements = [{"id": "test_id", "field_name": "test_field"}]
    query = "Test Query"
    result = alphageometry.semiose_vetter(actionable_elements, query)
    self.assertIn('id_result', result)

  def test_code_generator(self):
    website = "http://example.com"
    step_name = "Test Step"
    step_html_id = "test_id"
    prev_code = None
    model = "gpt-4"
    result = alphageometry.code_generator(website, step_name, step_html_id, prev_code, model)
    self.assertIn('response', result)

  def test_save_skill_to_memory(self):
    curriculum = "Test Curriculum"
    task = "Test Task"
    website = "http://example.com"
    curriculum_tokens = 100
    step_name = "Test Step"
    step_html_id = "test_id"
    step_tag_name = "test_tag"
    step_field_name = "test_field"
    step_duration = 1.0
    step_total_tokens = 50
    number_of_branches = 3
    step_code = "print('test')"
    model = "gpt-4"
    branch_stats = []
    alphageometry.save_skill_to_memory(curriculum, task, website, curriculum_tokens, step_name, step_html_id, step_tag_name, step_field_name, step_duration, step_total_tokens, number_of_branches, step_code, model, branch_stats)
    self.assertTrue(len(alphageometry.SKILL_MEMORY) > 0)

if __name__ == '__main__':
  absltest.main()
