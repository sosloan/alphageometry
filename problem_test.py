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

"""Unit tests for problem.py."""
import unittest

from absl.testing import absltest
import problem as pr


class ProblemTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)

  def test_orthocenter_no_translate(self):
    txt = 'a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c'  # pylint: disable=line-too-long

    # read the txt into pr.Problem object, do not change the name of points:
    p = pr.Problem.from_txt(txt, translate=False)

    # This is fed into the LM, translating from constructive to constrained:
    setup_str = p.setup_str_from_problem(ProblemTest.defs)

    self.assertEqual(
        setup_str,
        '{S} a : ; b : ; c : ; h : T a b c h 00 T a c b h 01 ? T a h b c',
    )

  def test_orthocenter_translate(self):
    txt = 'a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c'  # pylint: disable=line-too-long

    # Read the txt into pr.Problem object, change h -> d to match
    # training data distribution.
    p = pr.Problem.from_txt(txt, translate=True)

    # This is fed into the LM, translating from constructive to constrained:
    setup_str = p.setup_str_from_problem(ProblemTest.defs)

    self.assertEqual(
        setup_str,
        '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c',
    )

  def test_reshape(self):
    l = [1, 2, 3, 4, 5, 6]
    n = 2
    result = pr.reshape(l, n)
    self.assertEqual(result, [(1, 2), (3, 4), (5, 6)])

  def test_isint(self):
    self.assertTrue(pr.isint('123'))
    self.assertFalse(pr.isint('abc'))

  def test_construction_translate(self):
    construction = pr.Construction('test', ['a', 'b', 'c'])
    mapping = {'a': 'x', 'b': 'y'}
    translated = construction.translate(mapping)
    self.assertEqual(translated.name, 'test')
    self.assertEqual(translated.args, ['x', 'y', 'c'])

  def test_construction_txt(self):
    construction = pr.Construction('test', ['a', 'b', 'c'])
    self.assertEqual(construction.txt(), 'test a b c')

  def test_clause_translate(self):
    clause = pr.Clause(['a', 'b'], [pr.Construction('test', ['a', 'b', 'c'])])
    mapping = {'a': 'x', 'b': 'y'}
    translated = clause.translate(mapping)
    self.assertEqual(translated.points, ['x', 'y'])
    self.assertEqual(translated.constructions[0].args, ['x', 'y', 'c'])

  def test_clause_txt(self):
    clause = pr.Clause(['a', 'b'], [pr.Construction('test', ['a', 'b', 'c'])])
    self.assertEqual(clause.txt(), 'a b = test a b c')

  def test_simplify(self):
    self.assertEqual(pr.simplify(10, 5), (2, 1))
    self.assertEqual(pr.simplify(100, 25), (4, 1))

  def test_compare_fn(self):
    dep = pr.Dependency('test', ['a', 'b'], 'rule', 1)
    self.assertEqual(pr.compare_fn(dep), (dep, 'test a b'))

  def test_sort_deps(self):
    deps = [
        pr.Dependency('test2', ['a', 'b'], 'rule', 1),
        pr.Dependency('test1', ['a', 'b'], 'rule', 1),
    ]
    sorted_deps = pr.sort_deps(deps)
    self.assertEqual(sorted_deps[0].name, 'test1')
    self.assertEqual(sorted_deps[1].name, 'test2')

  def test_problem_copy(self):
    problem = pr.Problem('url', [pr.Clause(['a'], [])], pr.Construction('goal', ['a']))
    copied_problem = problem.copy()
    self.assertEqual(copied_problem.url, 'url')
    self.assertEqual(copied_problem.clauses[0].points, ['a'])
    self.assertEqual(copied_problem.goal.name, 'goal')

  def test_problem_txt(self):
    problem = pr.Problem('url', [pr.Clause(['a'], [pr.Construction('test', ['a'])])], pr.Construction('goal', ['a']))
    self.assertEqual(problem.txt(), 'a = test a ? goal a')

  def test_parse_rely(self):
    self.assertEqual(pr.parse_rely('a: b, c: d'), {'a': 'b', 'c': 'd'})
    self.assertEqual(pr.parse_rely(''), {})

  def test_definition_to_dict(self):
    definitions = [
        pr.Definition(pr.Construction('test1', ['a']), {}, pr.Clause([], []), [], []),
        pr.Definition(pr.Construction('test2', ['b']), {}, pr.Clause([], []), [], []),
    ]
    definitions_dict = pr.Definition.to_dict(definitions)
    self.assertEqual(definitions_dict['test1'].construction.name, 'test1')
    self.assertEqual(definitions_dict['test2'].construction.name, 'test2')

  def test_theorem_txt(self):
    theorem = pr.Theorem([pr.Construction('premise', ['a'])], [pr.Construction('conclusion', ['b'])])
    self.assertEqual(theorem.txt(), 'premise a => conclusion b')

  def test_theorem_conclusion_name_args(self):
    theorem = pr.Theorem([pr.Construction('premise', ['a'])], [pr.Construction('conclusion', ['b'])])
    mapping = {'b': 'point'}
    self.assertEqual(theorem.conclusion_name_args(mapping), ('conclusion', ['point']))

  def test_dependency_hashed(self):
    dep = pr.Dependency('test', ['a', 'b'], 'rule', 1)
    self.assertEqual(dep.hashed(), ('test', 'a', 'b'))

  def test_hashed_txt(self):
    self.assertEqual(pr.hashed_txt('test', ['a', 'b']), ('test', 'a', 'b'))


if __name__ == '__main__':
  absltest.main()
