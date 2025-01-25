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

"""Unit tests for dd."""
import unittest

from absl.testing import absltest
import dd
import graph as gh
import problem as pr


MAX_LEVEL = 1000


class DDTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_imo_2022_p4_should_succeed(self):
    p = pr.Problem.from_txt(
        'a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m ='
        ' on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n'
        ' g2 b; c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b,'
        ' on_circle d g2 b; e = on_line e a c, on_line e b d; p = on_line p a'
        ' n, on_line p c d; q = on_line q b n, on_line q c d ? cong e p e q'
    )
    g, _ = gh.Graph.build_problem(p, DDTest.defs)
    goal_args = g.names2nodes(p.goal.args)

    success = False
    for level in range(MAX_LEVEL):
      added, _, _, _ = dd.bfs_one_level(g, DDTest.rules, level, p)
      if g.check(p.goal.name, goal_args):
        success = True
        break
      if not added:  # saturated
        break

    self.assertTrue(success)

  def test_incenter_excenter_should_fail(self):
    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d = incenter d a b c; e = excenter e a b c ?'
        ' perp d c c e'
    )
    g, _ = gh.Graph.build_problem(p, DDTest.defs)
    goal_args = g.names2nodes(p.goal.args)

    success = False
    for level in range(MAX_LEVEL):
      added, _, _, _ = dd.bfs_one_level(g, DDTest.rules, level, p)
      if g.check(p.goal.name, goal_args):
        success = True
        break
      if not added:  # saturated
        break

    self.assertFalse(success)

  def test_match_eqratio_eqratio_eqratio(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqratio_eqratio_eqratio',
        premise=[
            pr.Clause(name='eqratio', args=['a', 'b', 'c', 'd', 'm', 'n', 'p', 'q']),
            pr.Clause(name='eqratio', args=['c', 'd', 'e', 'f', 'p', 'q', 'r', 'u']),
        ],
        conclusion=pr.Clause(name='eqratio', args=['a', 'b', 'e', 'f', 'm', 'n', 'r', 'u']),
    )
    matches = list(dd.match_eqratio_eqratio_eqratio(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle_eqangle_eqangle(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle_eqangle_eqangle',
        premise=[
            pr.Clause(name='eqangle', args=['a', 'b', 'c', 'd', 'm', 'n', 'p', 'q']),
            pr.Clause(name='eqangle', args=['c', 'd', 'e', 'f', 'p', 'q', 'r', 'u']),
        ],
        conclusion=pr.Clause(name='eqangle', args=['a', 'b', 'e', 'f', 'm', 'n', 'r', 'u']),
    )
    matches = list(dd.match_eqangle_eqangle_eqangle(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_perp_perp_npara_eqangle(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='perp_perp_npara_eqangle',
        premise=[
            pr.Clause(name='perp', args=['A', 'B', 'C', 'D']),
            pr.Clause(name='perp', args=['E', 'F', 'G', 'H']),
            pr.Clause(name='npara', args=['A', 'B', 'E', 'F']),
        ],
        conclusion=pr.Clause(name='eqangle', args=['A', 'B', 'E', 'F', 'C', 'D', 'G', 'H']),
    )
    matches = list(dd.match_perp_perp_npara_eqangle(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_circle_coll_eqangle_midp(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='circle_coll_eqangle_midp',
        premise=[
            pr.Clause(name='circle', args=['O', 'A', 'B', 'C']),
            pr.Clause(name='coll', args=['M', 'B', 'C']),
            pr.Clause(name='eqangle', args=['A', 'B', 'A', 'C', 'O', 'B', 'O', 'M']),
        ],
        conclusion=pr.Clause(name='midp', args=['M', 'B', 'C']),
    )
    matches = list(dd.match_circle_coll_eqangle_midp(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_midp_perp_cong(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='midp_perp_cong',
        premise=[
            pr.Clause(name='midp', args=['M', 'A', 'B']),
            pr.Clause(name='perp', args=['O', 'M', 'A', 'B']),
        ],
        conclusion=pr.Clause(name='cong', args=['O', 'A', 'O', 'B']),
    )
    matches = list(dd.match_midp_perp_cong(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_cyclic_eqangle_cong(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='cyclic_eqangle_cong',
        premise=[
            pr.Clause(name='cyclic', args=['A', 'B', 'C', 'P', 'Q', 'R']),
            pr.Clause(name='eqangle', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
        ],
        conclusion=pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
    )
    matches = list(dd.match_cyclic_eqangle_cong(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_circle_eqangle_perp(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='circle_eqangle_perp',
        premise=[
            pr.Clause(name='circle', args=['O', 'A', 'B', 'C']),
            pr.Clause(name='eqangle', args=['A', 'X', 'A', 'B', 'C', 'A', 'C', 'B']),
        ],
        conclusion=pr.Clause(name='perp', args=['O', 'A', 'A', 'X']),
    )
    matches = list(dd.match_circle_eqangle_perp(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_circle_perp_eqangle(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='circle_perp_eqangle',
        premise=[
            pr.Clause(name='circle', args=['O', 'A', 'B', 'C']),
            pr.Clause(name='perp', args=['O', 'A', 'A', 'X']),
        ],
        conclusion=pr.Clause(name='eqangle', args=['A', 'X', 'A', 'B', 'C', 'A', 'C', 'B']),
    )
    matches = list(dd.match_circle_perp_eqangle(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_perp_perp_ncoll_para(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='perp_perp_ncoll_para',
        premise=[
            pr.Clause(name='perp', args=['A', 'B', 'C', 'D']),
            pr.Clause(name='perp', args=['C', 'D', 'E', 'F']),
            pr.Clause(name='ncoll', args=['A', 'B', 'E']),
        ],
        conclusion=pr.Clause(name='para', args=['A', 'B', 'E', 'F']),
    )
    matches = list(dd.match_perp_perp_ncoll_para(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_ncoll_cong(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_ncoll_cong',
        premise=[
            pr.Clause(name='eqangle6', args=['A', 'O', 'A', 'B', 'B', 'A', 'B', 'O']),
            pr.Clause(name='ncoll', args=['O', 'A', 'B']),
        ],
        conclusion=pr.Clause(name='cong', args=['O', 'A', 'O', 'B']),
    )
    matches = list(dd.match_eqangle6_ncoll_cong(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle_perp_perp(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle_perp_perp',
        premise=[
            pr.Clause(name='eqangle', args=['A', 'B', 'P', 'Q', 'C', 'D', 'U', 'V']),
            pr.Clause(name='perp', args=['P', 'Q', 'U', 'V']),
        ],
        conclusion=pr.Clause(name='perp', args=['A', 'B', 'C', 'D']),
    )
    matches = list(dd.match_eqangle_perp_perp(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle_ncoll_cyclic(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle_ncoll_cyclic',
        premise=[
            pr.Clause(name='eqangle6', args=['P', 'A', 'P', 'B', 'Q', 'A', 'Q', 'B']),
            pr.Clause(name='ncoll', args=['P', 'Q', 'A', 'B']),
        ],
        conclusion=pr.Clause(name='cyclic', args=['A', 'B', 'P', 'Q']),
    )
    matches = list(dd.match_eqangle_ncoll_cyclic(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle_para(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle_para',
        premise=[
            pr.Clause(name='eqangle', args=['A', 'B', 'P', 'Q', 'C', 'D', 'P', 'Q']),
        ],
        conclusion=pr.Clause(name='para', args=['A', 'B', 'C', 'D']),
    )
    matches = list(dd.match_eqangle_para(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_cyclic_eqangle(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='cyclic_eqangle',
        premise=[
            pr.Clause(name='cyclic', args=['A', 'B', 'P', 'Q']),
        ],
        conclusion=pr.Clause(name='eqangle', args=['P', 'A', 'P', 'B', 'Q', 'A', 'Q', 'B']),
    )
    matches = list(dd.match_cyclic_eqangle(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_cong_cong_cong_cyclic(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='cong_cong_cong_cyclic',
        premise=[
            pr.Clause(name='cong', args=['O', 'A', 'O', 'B']),
            pr.Clause(name='cong', args=['O', 'B', 'O', 'C']),
            pr.Clause(name='cong', args=['O', 'C', 'O', 'D']),
        ],
        conclusion=pr.Clause(name='cyclic', args=['A', 'B', 'C', 'D']),
    )
    matches = list(dd.match_cong_cong_cong_cyclic(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_cong_cong_cong_ncoll_contri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='cong_cong_cong_ncoll_contri',
        premise=[
            pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
            pr.Clause(name='cong', args=['B', 'C', 'Q', 'R']),
            pr.Clause(name='cong', args=['C', 'A', 'R', 'P']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='contri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_cong_cong_cong_ncoll_contri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_cong_cong_eqangle6_ncoll_contri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='cong_cong_eqangle6_ncoll_contri',
        premise=[
            pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
            pr.Clause(name='cong', args=['B', 'C', 'Q', 'R']),
            pr.Clause(name='eqangle6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='contri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_cong_cong_eqangle6_ncoll_contri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqratio6_eqangle6_ncoll_simtri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqratio6_eqangle6_ncoll_simtri',
        premise=[
            pr.Clause(name='eqratio6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='eqangle6', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='simtri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqratio6_eqangle6_ncoll_simtri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_eqangle6_ncoll_simtri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_eqangle6_ncoll_simtri',
        premise=[
            pr.Clause(name='eqangle6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='eqangle6', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='simtri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqangle6_eqangle6_ncoll_simtri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqratio6_eqratio6_ncoll_simtri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqratio6_eqratio6_ncoll_simtri',
        premise=[
            pr.Clause(name='eqratio6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='eqratio6', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='simtri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqratio6_eqratio6_ncoll_simtri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_eqangle6_ncoll_simtri2(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_eqangle6_ncoll_simtri2',
        premise=[
            pr.Clause(name='eqangle6', args=['B', 'A', 'B', 'C', 'Q', 'R', 'Q', 'P']),
            pr.Clause(name='eqangle6', args=['C', 'A', 'C', 'B', 'R', 'Q', 'R', 'P']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
        ],
        conclusion=pr.Clause(name='simtri2', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqangle6_eqangle6_ncoll_simtri2(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqratio6_eqratio6_ncoll_cong_contri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqratio6_eqratio6_ncoll_cong_contri',
        premise=[
            pr.Clause(name='eqratio6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='eqratio6', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
            pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
        ],
        conclusion=pr.Clause(name='contri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqratio6_eqratio6_ncoll_cong_contri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_eqangle6_ncoll_cong_contri(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_eqangle6_ncoll_cong_contri',
        premise=[
            pr.Clause(name='eqangle6', args=['B', 'A', 'B', 'C', 'Q', 'P', 'Q', 'R']),
            pr.Clause(name='eqangle6', args=['C', 'A', 'C', 'B', 'R', 'P', 'R', 'Q']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
            pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
        ],
        conclusion=pr.Clause(name='contri', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqangle6_eqangle6_ncoll_cong_contri(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_eqangle6_ncoll_cong_contri2(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_eqangle6_ncoll_cong_contri2',
        premise=[
            pr.Clause(name='eqangle6', args=['B', 'A', 'B', 'C', 'Q', 'R', 'Q', 'P']),
            pr.Clause(name='eqangle6', args=['C', 'A', 'C', 'B', 'R', 'Q', 'R', 'P']),
            pr.Clause(name='ncoll', args=['A', 'B', 'C']),
            pr.Clause(name='cong', args=['A', 'B', 'P', 'Q']),
        ],
        conclusion=pr.Clause(name='contri2', args=['A', 'B', 'C', 'P', 'Q', 'R']),
    )
    matches = list(dd.match_eqangle6_eqangle6_ncoll_cong_contri2(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqratio6_coll_ncoll_eqangle6(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqratio6_coll_ncoll_eqangle6',
        premise=[
            pr.Clause(name='eqratio6', args=['d', 'b', 'd', 'c', 'a', 'b', 'a', 'c']),
            pr.Clause(name='coll', args=['d', 'b', 'c']),
            pr.Clause(name='ncoll', args=['a', 'b', 'c']),
        ],
        conclusion=pr.Clause(name='eqangle6', args=['a', 'b', 'a', 'd', 'a', 'd', 'a', 'c']),
    )
    matches = list(dd.match_eqratio6_coll_ncoll_eqangle6(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_coll_ncoll_eqratio6(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_coll_ncoll_eqratio6',
        premise=[
            pr.Clause(name='eqangle6', args=['a', 'b', 'a', 'd', 'a', 'd', 'a', 'c']),
            pr.Clause(name='coll', args=['d', 'b', 'c']),
            pr.Clause(name='ncoll', args=['a', 'b', 'c']),
        ],
        conclusion=pr.Clause(name='eqratio6', args=['d', 'b', 'd', 'c', 'a', 'b', 'a', 'c']),
    )
    matches = list(dd.match_eqangle6_coll_ncoll_eqratio6(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_eqangle6_ncoll_cyclic(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle6_ncoll_cyclic',
        premise=[
            pr.Clause(name='eqangle6', args=['P', 'A', 'P', 'B', 'Q', 'A', 'Q', 'B']),
            pr.Clause(name='ncoll', args=['P', 'Q', 'A', 'B']),
        ],
        conclusion=pr.Clause(name='cyclic', args=['A', 'B', 'P', 'Q']),
    )
    matches = list(dd.match_eqangle6_ncoll_cyclic(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_all(self):
    g = gh.Graph()
    matches = list(dd.match_all('coll', g))
    self.assertIsInstance(matches, list)

  def test_cache_match(self):
    g = gh.Graph()
    cache = dd.cache_match(g)
    matches = cache('coll')
    self.assertIsInstance(matches, list)

  def test_try_to_map(self):
    clause_enum = [
        (pr.Clause(name='coll', args=['a', 'b', 'c']), [('a', 'b', 'c')]),
        (pr.Clause(name='coll', args=['d', 'e', 'f']), [('d', 'e', 'f')]),
    ]
    mapping = {'a': 'a', 'b': 'b', 'c': 'c'}
    mappings = list(dd.try_to_map(clause_enum, mapping))
    self.assertIsInstance(mappings, list)

  def test_match_generic(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='generic',
        premise=[
            pr.Clause(name='coll', args=['a', 'b', 'c']),
            pr.Clause(name='coll', args=['d', 'e', 'f']),
        ],
        conclusion=pr.Clause(name='coll', args=['a', 'b', 'c']),
    )
    matches = list(dd.match_generic(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_BUILT_IN_FNS(self):
    self.assertIsInstance(dd.BUILT_IN_FNS, dict)

  def test_set_skip_theorems(self):
    dd.set_skip_theorems({'theorem1', 'theorem2'})
    self.assertIn('theorem1', dd.SKIP_THEOREMS)
    self.assertIn('theorem2', dd.SKIP_THEOREMS)

  def test_MAX_BRANCH(self):
    self.assertIsInstance(dd.MAX_BRANCH, int)

  def test_match_one_theorem(self):
    g = gh.Graph()
    theorem = pr.Theorem(
        name='eqangle_eqangle_eqangle',
        premise=[
            pr.Clause(name='eqangle', args=['a', 'b', 'c', 'd', 'm', 'n', 'p', 'q']),
            pr.Clause(name='eqangle', args=['c', 'd', 'e', 'f', 'p', 'q', 'r', 'u']),
        ],
        conclusion=pr.Clause(name='eqangle', args=['a', 'b', 'e', 'f', 'm', 'n', 'r', 'u']),
    )
    matches = list(dd.match_one_theorem(g, dd.cache_match(g), theorem))
    self.assertIsInstance(matches, list)

  def test_match_all_theorems(self):
    g = gh.Graph()
    theorems = {
        'theorem1': pr.Theorem(
            name='theorem1',
            premise=[
                pr.Clause(name='coll', args=['a', 'b', 'c']),
            ],
            conclusion=pr.Clause(name='coll', args=['a', 'b', 'c']),
        ),
        'theorem2': pr.Theorem(
            name='theorem2',
            premise=[
                pr.Clause(name='coll', args=['d', 'e', 'f']),
            ],
            conclusion=pr.Clause(name='coll', args=['d', 'e', 'f']),
        ),
    }
    matches = dd.match_all_theorems(g, theorems, None)
    self.assertIsInstance(matches, dict)

  def test_bfs_one_level(self):
    g = gh.Graph()
    theorems = {
        'theorem1': pr.Theorem(
            name='theorem1',
            premise=[
                pr.Clause(name='coll', args=['a', 'b', 'c']),
            ],
            conclusion=pr.Clause(name='coll', args=['a', 'b', 'c']),
        ),
    }
    level = 0
    controller = pr.Problem(goal=pr.Clause(name='coll', args=['a', 'b', 'c']))
    added, derives, eq4s, branching = dd.bfs_one_level(g, theorems, level, controller)
    self.assertIsInstance(added, list)
    self.assertIsInstance(derives, dict)
    self.assertIsInstance(eq4s, dict)
    self.assertIsInstance(branching, int)

  def test_create_consts_str(self):
    g = gh.Graph()
    result = dd.create_consts_str(g, 'pi/2')
    self.assertIsInstance(result, gh.Angle)

  def test_do_algebra(self):
    g = gh.Graph()
    dep = pr.Dependency(name='para', args=['a', 'b'])
    result = dd.do_algebra(g, dep, False)
    self.assertIsInstance(result, list)

  def test_apply_derivations(self):
    g = gh.Graph()
    derives = {'eqangle': [(gh.Point('a'), gh.Point('b'), gh.Point('c'), gh.Point('d'))]}
    result = dd.apply_derivations(g, derives)
    self.assertIsInstance(result, list)


if __name__ == '__main__':
  absltest.main()
