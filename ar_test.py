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

"""Unit tests for ar.py."""
import unittest

from absl.testing import absltest
import ar
import graph as gh
import problem as pr


class ARTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)
    cls.rules = pr.Theorem.from_txt_file('rules.txt', to_dict=True)

  def test_update_groups(self):
    """Test for update_groups."""
    groups1 = [{1, 2}, {3, 4, 5}, {6, 7}]
    groups2 = [{2, 3, 8}, {9, 10, 11}]

    _, links, history = ar.update_groups(groups1, groups2)
    self.assertEqual(
        history,
        [
            [{1, 2, 3, 4, 5, 8}, {6, 7}],
            [{1, 2, 3, 4, 5, 8}, {6, 7}, {9, 10, 11}],
        ],
    )
    self.assertEqual(links, [(2, 3), (3, 8), (9, 10), (10, 11)])

    groups1 = [{1, 2}, {3, 4}, {5, 6}, {7, 8}]
    groups2 = [{2, 3, 8, 9, 10}, {3, 6, 11}]

    _, links, history = ar.update_groups(groups1, groups2)
    self.assertEqual(
        history,
        [
            [{1, 2, 3, 4, 7, 8, 9, 10}, {5, 6}],
            [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}],
        ],
    )
    self.assertEqual(links, [(2, 3), (3, 8), (8, 9), (9, 10), (3, 6), (6, 11)])

    groups1 = []
    groups2 = [{1, 2}, {3, 4}, {5, 6}, {2, 3}]

    _, links, history = ar.update_groups(groups1, groups2)
    self.assertEqual(
        history,
        [
            [{1, 2}],
            [{1, 2}, {3, 4}],
            [{1, 2}, {3, 4}, {5, 6}],
            [{1, 2, 3, 4}, {5, 6}],
        ],
    )
    self.assertEqual(links, [(1, 2), (3, 4), (5, 6), (2, 3)])

  def test_generic_table_simple(self):
    tb = ar.Table()

    # If a-b = b-c & d-a = c-d
    tb.add_eq4('a', 'b', 'b', 'c', 'fact1')
    tb.add_eq4('d', 'a', 'c', 'd', 'fact2')
    tb.add_eq4('x', 'y', 'z', 't', 'fact3')  # distractor fact

    # Then b=d, because {fact1, fact2} but not fact3.
    result = list(tb.get_all_eqs_and_why())
    self.assertIn(('b', 'd', ['fact1', 'fact2']), result)

  def test_angle_table_inbisector_exbisector(self):
    """Test that AR can figure out bisector & ex-bisector are perpendicular."""
    # Load the scenario that we have cd is bisector of acb and
    # ce is the ex-bisector of acb.
    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d = incenter d a b c; e = excenter e a b c ?'
        ' perp d c c e'
    )
    g, _ = gh.Graph.build_problem(p, ARTest.defs)

    # Create an external angle table:
    tb = ar.AngleTable('pi')

    # Add bisector & ex-bisector facts into the table:
    ca, cd, cb, ce = g.names2nodes(['d(ac)', 'd(cd)', 'd(bc)', 'd(ce)'])
    tb.add_eqangle(ca, cd, cd, cb, 'fact1')
    tb.add_eqangle(ce, ca, cb, ce, 'fact2')

    # Add a distractor fact to make sure traceback does not include this fact
    ab = g.names2nodes(['d(ab)'])[0]
    tb.add_eqangle(ab, cb, cb, ca, 'fact3')

    # Check for all new equalities
    result = list(tb.get_all_eqs_and_why())

    # halfpi is represented as a tuple (1, 2)
    halfpi = (1, 2)

    # check that cd-ce == halfpi and this is because fact1 & fact2, not fact3
    self.assertCountEqual(
        result,
        [
            (cd, ce, halfpi, ['fact1', 'fact2']),
            (ce, cd, halfpi, ['fact1', 'fact2']),
        ],
    )

  def test_angle_table_equilateral_triangle(self):
    """Test that AR can figure out triangles with 3 equal angles => each is pi/3."""
    # Load an equaliteral scenario
    p = pr.Problem.from_txt('a b c = ieq_triangle ? cong a b a c')
    g, _ = gh.Graph.build_problem(p, ARTest.defs)

    # Add two eqangles facts because ieq_triangle only add congruent sides
    a, b, c = g.names2nodes('abc')
    g.add_eqangle([a, b, b, c, b, c, c, a], pr.EmptyDependency(0, None))
    g.add_eqangle([b, c, c, a, c, a, a, b], pr.EmptyDependency(0, None))

    # Create an external angle table:
    tb = ar.AngleTable('pi')

    # Add the fact that there are three equal angles
    ab, bc, ca = g.names2nodes(['d(ab)', 'd(bc)', 'd(ac)'])
    tb.add_eqangle(ab, bc, bc, ca, 'fact1')
    tb.add_eqangle(bc, ca, ca, ab, 'fact2')

    # Now check for all new equalities
    result = list(tb.get_all_eqs_and_why())
    result = [(x.name, y.name, z, t) for x, y, z, t in result]

    # 1/3 pi is represented as a tuple angle_60
    angle_60 = (1, 3)
    angle_120 = (2, 3)

    # check that angles constants are created and figured out:
    self.assertCountEqual(
        result,
        [
            ('d(bc)', 'd(ac)', angle_120, ['fact1', 'fact2']),
            ('d(ab)', 'd(bc)', angle_120, ['fact1', 'fact2']),
            ('d(ac)', 'd(ab)', angle_120, ['fact1', 'fact2']),
            ('d(ac)', 'd(bc)', angle_60, ['fact1', 'fact2']),
            ('d(bc)', 'd(ab)', angle_60, ['fact1', 'fact2']),
            ('d(ab)', 'd(ac)', angle_60, ['fact1', 'fact2']),
        ],
    )

  def test_incenter_excenter_touchpoints(self):
    """Test that AR can figure out incenter/excenter touchpoints are equidistant to midpoint."""

    p = pr.Problem.from_txt(
        'a b c = triangle a b c; d1 d2 d3 d = incenter2 a b c; e1 e2 e3 e ='
        ' excenter2 a b c ? perp d c c e',
        translate=False,
    )
    g, _ = gh.Graph.build_problem(p, ARTest.defs)

    a, b, c, ab, bc, ca, d1, d2, d3, e1, e2, e3 = g.names2nodes(
        ['a', 'b', 'c', 'ab', 'bc', 'ac', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3']
    )

    # Create an external distance table:
    tb = ar.DistanceTable()

    # DD can figure out the following facts,
    # we manually add them to AR.
    tb.add_cong(ab, ca, a, d3, a, d2, 'fact1')
    tb.add_cong(ab, ca, a, e3, a, e2, 'fact2')
    tb.add_cong(ca, bc, c, d2, c, d1, 'fact5')
    tb.add_cong(ca, bc, c, e2, c, e1, 'fact6')
    tb.add_cong(bc, ab, b, d1, b, d3, 'fact3')
    tb.add_cong(bc, ab, b, e1, b, e3, 'fact4')

    # Now we check whether tb has figured out that
    # distance(b, d1) == distance(e1, c)

    # linear comb exprssion of each variables:
    b = tb.v2e['bc:b']
    c = tb.v2e['bc:c']
    d1 = tb.v2e['bc:d1']
    e1 = tb.v2e['bc:e1']

    self.assertEqual(ar.minus(d1, b), ar.minus(c, e1))

  def test_simplify(self):
    self.assertEqual(ar.simplify(10, 5), (2, 1))
    self.assertEqual(ar.simplify(100, 25), (4, 1))
    self.assertEqual(ar.simplify(8, 4), (2, 1))
    self.assertEqual(ar.simplify(9, 3), (3, 1))

  def test_get_quotient(self):
    self.assertEqual(ar.get_quotient(0.5), (1, 2))
    self.assertEqual(ar.get_quotient(0.25), (1, 4))
    self.assertEqual(ar.get_quotient(0.75), (3, 4))
    self.assertEqual(ar.get_quotient(0.2), (1, 5))

  def test_fix_v(self):
    self.assertEqual(ar.fix_v(0.5), 0.5)
    self.assertEqual(ar.fix_v(0.25), 0.25)
    self.assertEqual(ar.fix_v(0.75), 0.75)
    self.assertEqual(ar.fix_v(0.2), 0.2)

  def test_fix(self):
    self.assertEqual(ar.fix({'a': 0.5, 'b': 0.25}), {'a': 0.5, 'b': 0.25})
    self.assertEqual(ar.fix({'a': 0.75, 'b': 0.2}), {'a': 0.75, 'b': 0.2})

  def test_frac_string(self):
    self.assertEqual(ar.frac_string(0.5), '1/2')
    self.assertEqual(ar.frac_string(0.25), '1/4')
    self.assertEqual(ar.frac_string(0.75), '3/4')
    self.assertEqual(ar.frac_string(0.2), '1/5')

  def test_hashed(self):
    self.assertEqual(ar.hashed({'a': 0.5, 'b': 0.25}), (('a', 0.5), ('b', 0.25)))
    self.assertEqual(ar.hashed({'a': 0.75, 'b': 0.2}), (('a', 0.75), ('b', 0.2)))

  def test_is_zero(self):
    self.assertTrue(ar.is_zero({'a': 0, 'b': 0}))
    self.assertFalse(ar.is_zero({'a': 0.5, 'b': 0.25}))

  def test_strip(self):
    self.assertEqual(ar.strip({'a': 0, 'b': 0.25}), {'b': 0.25})
    self.assertEqual(ar.strip({'a': 0.5, 'b': 0}), {'a': 0.5})

  def test_plus(self):
    self.assertEqual(ar.plus({'a': 0.5}, {'b': 0.25}), {'a': 0.5, 'b': 0.25})
    self.assertEqual(ar.plus({'a': 0.5}, {'a': 0.25}), {'a': 0.75})

  def test_plus_all(self):
    self.assertEqual(
        ar.plus_all({'a': 0.5}, {'b': 0.25}, {'c': 0.75}),
        {'a': 0.5, 'b': 0.25, 'c': 0.75},
    )
    self.assertEqual(
        ar.plus_all({'a': 0.5}, {'a': 0.25}, {'a': 0.75}), {'a': 1.5}
    )

  def test_mult(self):
    self.assertEqual(ar.mult({'a': 0.5, 'b': 0.25}, 2), {'a': 1.0, 'b': 0.5})
    self.assertEqual(ar.mult({'a': 0.5, 'b': 0.25}, 0.5), {'a': 0.25, 'b': 0.125})

  def test_minus(self):
    self.assertEqual(ar.minus({'a': 0.5}, {'b': 0.25}), {'a': 0.5, 'b': -0.25})
    self.assertEqual(ar.minus({'a': 0.5}, {'a': 0.25}), {'a': 0.25})

  def test_div(self):
    self.assertEqual(ar.div({'a': 0.5}, {'a': 0.25}), 2.0)
    self.assertEqual(ar.div({'a': 0.5, 'b': 0.25}, {'a': 0.25, 'b': 0.125}), 2.0)

  def test_recon(self):
    self.assertEqual(ar.recon({'a': 0.5, 'b': 0.25}, 'a'), ('b', {'a': -2.0}))
    self.assertEqual(ar.recon({'a': 0.5, 'b': 0.25}, 'b'), ('a', {'b': -0.5}))

  def test_replace(self):
    self.assertEqual(
        ar.replace({'a': 0.5, 'b': 0.25}, 'a', {'c': 0.75}),
        {'b': 0.25, 'c': 0.375},
    )
    self.assertEqual(
        ar.replace({'a': 0.5, 'b': 0.25}, 'b', {'c': 0.75}),
        {'a': 0.5, 'c': 0.1875},
    )

  def test_comb2(self):
    self.assertEqual(list(ar.comb2([1, 2, 3])), [(1, 2), (1, 3), (2, 3)])
    self.assertEqual(list(ar.comb2([1, 2])), [(1, 2)])
    self.assertEqual(list(ar.comb2([1])), [])

  def test_perm2(self):
    self.assertEqual(
        list(ar.perm2([1, 2, 3])),
        [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)],
    )
    self.assertEqual(list(ar.perm2([1, 2])), [(1, 2), (2, 1)])
    self.assertEqual(list(ar.perm2([1])), [])

  def test_chain2(self):
    self.assertEqual(list(ar.chain2([1, 2, 3])), [(1, 2), (2, 3)])
    self.assertEqual(list(ar.chain2([1, 2])), [(1, 2)])
    self.assertEqual(list(ar.chain2([1])), [])

  def test_table_add_free(self):
    tb = ar.Table()
    tb.add_free('a')
    self.assertIn('a', tb.v2e)

  def test_table_replace(self):
    tb = ar.Table()
    tb.add_free('a')
    tb.add_free('b')
    tb.replace('a', {'b': 0.5})
    self.assertEqual(tb.v2e['b'], {'b': 1.0})
    self.assertEqual(tb.v2e['a'], {'b': 0.5})

  def test_table_add_expr(self):
    tb = ar.Table()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_register(self):
    tb = ar.Table()
    tb.register([('a', 0.5), ('b', 0.25)], 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_register2(self):
    tb = ar.Table()
    tb.register2('a', 'b', 0.5, 0.25, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_register3(self):
    tb = ar.Table()
    tb.register3('a', 'b', 0.5, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_register4(self):
    tb = ar.Table()
    tb.register4('a', 'b', 'c', 'd', 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_why(self):
    tb = ar.Table()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(tb.why({'a': 0.5}), [])

  def test_table_record_eq(self):
    tb = ar.Table()
    tb.record_eq('a', 'b', 'c', 'd')
    self.assertIn(('a', 'b', 'c', 'd'), tb.eqs)

  def test_table_check_record_eq(self):
    tb = ar.Table()
    tb.record_eq('a', 'b', 'c', 'd')
    self.assertTrue(tb.check_record_eq('a', 'b', 'c', 'd'))
    self.assertFalse(tb.check_record_eq('a', 'b', 'c', 'e'))

  def test_table_add_eq2(self):
    tb = ar.Table()
    tb.add_eq2('a', 'b', 0.5, 0.25, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_add_eq3(self):
    tb = ar.Table()
    tb.add_eq3('a', 'b', 0.5, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_add_eq4(self):
    tb = ar.Table()
    tb.add_eq4('a', 'b', 'c', 'd', 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_table_pairs(self):
    tb = ar.Table()
    tb.add_free('a')
    tb.add_free('b')
    self.assertEqual(list(tb.pairs()), [('a', 'b'), ('b', 'a')])

  def test_table_modulo(self):
    tb = ar.Table()
    self.assertEqual(tb.modulo({'a': 0.5}), {'a': 0.5})

  def test_table_get_all_eqs(self):
    tb = ar.Table()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(tb.get_all_eqs(), {(('a', 0.5),): [('a', 'b')]})

  def test_table_get_all_eqs_and_why(self):
    tb = ar.Table()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(list(tb.get_all_eqs_and_why()), [])

  def test_geometric_table_get_name(self):
    tb = ar.GeometricTable()
    self.assertEqual(tb.get_name([gh.Point('a')]), ['a'])

  def test_geometric_table_map2obj(self):
    tb = ar.GeometricTable()
    tb.get_name([gh.Point('a')])
    self.assertEqual(tb.map2obj(['a']), [gh.Point('a')])

  def test_geometric_table_get_all_eqs_and_why(self):
    tb = ar.GeometricTable()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(list(tb.get_all_eqs_and_why(True)), [])

  def test_ratio_table_add_eq(self):
    tb = ar.RatioTable()
    tb.add_eq(gh.Length('a'), gh.Length('b'), 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -1.0})

  def test_ratio_table_add_const_ratio(self):
    tb = ar.RatioTable()
    tb.add_const_ratio(gh.Length('a'), gh.Length('b'), 0.5, 0.25, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_ratio_table_add_eqratio(self):
    tb = ar.RatioTable()
    tb.add_eqratio(
        gh.Length('a'),
        gh.Length('b'),
        gh.Length('c'),
        gh.Length('d'),
        'fact1',
    )
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -1.0})

  def test_ratio_table_get_all_eqs_and_why(self):
    tb = ar.RatioTable()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(list(tb.get_all_eqs_and_why(True)), [])

  def test_angle_table_modulo(self):
    tb = ar.AngleTable()
    self.assertEqual(tb.modulo({'a': 0.5}), {'a': 0.5})

  def test_angle_table_add_para(self):
    tb = ar.AngleTable()
    tb.add_para(gh.Direction('a'), gh.Direction('b'), 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -1.0})

  def test_angle_table_add_const_angle(self):
    tb = ar.AngleTable()
    tb.add_const_angle(gh.Direction('a'), gh.Direction('b'), 0.5, 'fact1')
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -2.0})

  def test_angle_table_add_eqangle(self):
    tb = ar.AngleTable()
    tb.add_eqangle(
        gh.Direction('a'),
        gh.Direction('b'),
        gh.Direction('c'),
        gh.Direction('d'),
        'fact1',
    )
    self.assertEqual(tb.v2e['a'], {'a': 1.0})
    self.assertEqual(tb.v2e['b'], {'a': -1.0})

  def test_angle_table_get_all_eqs_and_why(self):
    tb = ar.AngleTable()
    tb.add_expr([('a', 0.5), ('b', 0.25)])
    self.assertEqual(list(tb.get_all_eqs_and_why(True)), [])

  def test_distance_table_pairs(self):
    tb = ar.DistanceTable()
    tb.add_free('a:1')
    tb.add_free('a:2')
    self.assertEqual(list(tb.pairs()), [('a:1', 'a:2'), ('a:2', 'a:1')])

  def test_distance_table_name(self):
    tb = ar.DistanceTable()
    self.assertEqual(tb.name(gh.Line('a'), gh.Point('b')), 'a:b')

  def test_distance_table_map2obj(self):
    tb = ar.DistanceTable()
    tb.name(gh.Line('a'), gh.Point('b'))
    self.assertEqual(tb.map2obj(['a:b']), [gh.Point('b')])

  def test_distance_table_add_cong(self):
    tb = ar.DistanceTable()
    tb.add_cong(
        gh.Line('a'),
        gh.Line('b'),
        gh.Point('c'),
        gh.Point('d'),
        gh.Point('e'),
        gh.Point('f'),
        'fact1',
    )
    self.assertEqual(tb.v2e['a:c'], {'a:c': 1.0})
    self.assertEqual(tb.v2e['a:d'], {'a:c': -1.0})

  def test_distance_table_get_all_eqs_and_why(self):
    tb = ar.DistanceTable()
    tb.add_expr([('a:1', 0.5), ('a:2', 0.25)])
    self.assertEqual(list(tb.get_all_eqs_and_why(True)), [])
