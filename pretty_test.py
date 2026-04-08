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

"""Unit tests for pretty.py."""

import unittest

from absl.testing import absltest
import pretty


class MapSymbolTest(unittest.TestCase):

  def test_map_symbol_all_keys(self):
    expected = {
        'T': 'perp',
        'P': 'para',
        'D': 'cong',
        'S': 'simtri',
        'I': 'circle',
        'M': 'midp',
        'O': 'cyclic',
        'C': 'coll',
        '^': 'eqangle',
        '/': 'eqratio',
        '%': 'eqratio',
        '=': 'contri',
        'X': 'collx',
        'A': 'acompute',
        'R': 'rcompute',
        'Q': 'fixc',
        'E': 'fixl',
        'V': 'fixb',
        'H': 'fixt',
        'Z': 'fixp',
        'Y': 'ind',
    }
    for symbol, name in expected.items():
      self.assertEqual(pretty.map_symbol(symbol), name)

  def test_map_symbol_inv_roundtrip(self):
    # For symbols with unique values, map_symbol_inv(map_symbol(c)) == c
    unique_symbols = [
        'T', 'P', 'D', 'S', 'I', 'M', 'O', 'C',
        '^', '=', 'X', 'A', 'R', 'Q', 'E', 'V', 'H', 'Z', 'Y',
    ]
    for symbol in unique_symbols:
      name = pretty.map_symbol(symbol)
      self.assertEqual(pretty.map_symbol_inv(name), symbol)

  def test_map_symbol_inv_known_names(self):
    self.assertEqual(pretty.map_symbol_inv('perp'), 'T')
    self.assertEqual(pretty.map_symbol_inv('para'), 'P')
    self.assertEqual(pretty.map_symbol_inv('cong'), 'D')
    self.assertEqual(pretty.map_symbol_inv('coll'), 'C')
    self.assertEqual(pretty.map_symbol_inv('cyclic'), 'O')
    self.assertEqual(pretty.map_symbol_inv('midp'), 'M')


class SimplifyTest(unittest.TestCase):

  def test_simplify_already_reduced(self):
    self.assertEqual(pretty.simplify(1, 3), (1, 3))
    self.assertEqual(pretty.simplify(2, 3), (2, 3))

  def test_simplify_common_factor(self):
    self.assertEqual(pretty.simplify(6, 4), (3, 2))
    self.assertEqual(pretty.simplify(12, 8), (3, 2))

  def test_simplify_one(self):
    self.assertEqual(pretty.simplify(5, 5), (1, 1))
    self.assertEqual(pretty.simplify(7, 1), (7, 1))

  def test_simplify_large(self):
    self.assertEqual(pretty.simplify(100, 75), (4, 3))


class Pretty2rTest(unittest.TestCase):

  def test_basic_no_swap(self):
    # No swaps needed when b not in (c, d) and a != d
    self.assertEqual(pretty.pretty2r('a', 'b', 'c', 'd'), 'a b c d')

  def test_b_equals_c_swap(self):
    # b == c: swap a and b
    self.assertEqual(pretty.pretty2r('a', 'c', 'c', 'd'), 'c a c d')

  def test_b_equals_d_swap(self):
    # b == d: swap a and b; then since new a ('d') == old d, also swap c and d
    self.assertEqual(pretty.pretty2r('a', 'd', 'c', 'd'), 'd a d c')

  def test_a_equals_d_swap_cd(self):
    # After first swap (if any), a == d: swap c and d
    self.assertEqual(pretty.pretty2r('a', 'b', 'd', 'a'), 'a b a d')


class Pretty2aTest(unittest.TestCase):

  def test_basic_no_swap(self):
    self.assertEqual(pretty.pretty2a('a', 'b', 'c', 'd'), 'a b c d')

  def test_b_equals_c_swap(self):
    self.assertEqual(pretty.pretty2a('a', 'c', 'c', 'd'), 'c a c d')

  def test_b_equals_d_swap(self):
    # b == d: swap a and b; then since new a ('d') == old d, also swap c and d
    self.assertEqual(pretty.pretty2a('a', 'd', 'c', 'd'), 'd a d c')


class PrettyAngleTest(unittest.TestCase):

  def test_angle_different_arms(self):
    # a != c: returns full angle notation
    result = pretty.pretty_angle('a', 'b', 'c', 'd')
    self.assertEqual(result, '\u2220(ab-cd)')

  def test_angle_same_vertex(self):
    # a == c: returns simple angle notation
    result = pretty.pretty_angle('a', 'b', 'a', 'd')
    self.assertEqual(result, '\u2220bad')

  def test_b_in_cd_swap_ab(self):
    # b in (c, d): swap a and b
    result = pretty.pretty_angle('a', 'c', 'c', 'd')
    self.assertIn('c', result)


class PrettyNlTest(unittest.TestCase):

  def test_coll(self):
    self.assertEqual(pretty.pretty_nl('coll', ['a', 'b', 'c']), 'a,b,c are collinear')

  def test_cyclic(self):
    self.assertEqual(pretty.pretty_nl('cyclic', ['a', 'b', 'c', 'd']), 'a,b,c,d are concyclic')

  def test_midp(self):
    self.assertEqual(pretty.pretty_nl('midp', ['x', 'a', 'b']), 'x is midpoint of ab')

  def test_midpoint_alias(self):
    self.assertEqual(pretty.pretty_nl('midpoint', ['x', 'a', 'b']), 'x is midpoint of ab')

  def test_cong(self):
    self.assertEqual(pretty.pretty_nl('cong', ['a', 'b', 'c', 'd']), 'ab = cd')

  def test_perp_four_args(self):
    result = pretty.pretty_nl('perp', ['a', 'b', 'c', 'd'])
    self.assertIn('ab', result)
    self.assertIn('cd', result)
    self.assertIn('\u27c2', result)

  def test_perp_two_args_algebraic(self):
    result = pretty.pretty_nl('perp', ['ab', 'cd'])
    self.assertIn('ab', result)
    self.assertIn('cd', result)
    self.assertIn('\u27c2', result)

  def test_para_four_args(self):
    result = pretty.pretty_nl('para', ['a', 'b', 'c', 'd'])
    self.assertIn('ab', result)
    self.assertIn('cd', result)
    self.assertIn('\u2225', result)

  def test_para_two_args_algebraic(self):
    result = pretty.pretty_nl('para', ['ab', 'cd'])
    self.assertIn('ab', result)
    self.assertIn('cd', result)
    self.assertIn('\u2225', result)

  def test_simtri(self):
    result = pretty.pretty_nl('simtri', ['a', 'b', 'c', 'x', 'y', 'z'])
    self.assertIn('\u0394abc', result)
    self.assertIn('\u0394xyz', result)
    self.assertIn('similar', result)

  def test_contri(self):
    result = pretty.pretty_nl('contri', ['a', 'b', 'c', 'x', 'y', 'z'])
    self.assertIn('\u0394abc', result)
    self.assertIn('\u0394xyz', result)
    self.assertIn('congruent', result)

  def test_circle(self):
    result = pretty.pretty_nl('circle', ['o', 'a', 'b', 'c'])
    self.assertIn('o', result)
    self.assertIn('circumcenter', result)

  def test_foot(self):
    result = pretty.pretty_nl('foot', ['a', 'b', 'c', 'd'])
    self.assertIn('a', result)
    self.assertIn('foot', result)

  def test_eqangle(self):
    result = pretty.pretty_nl('eqangle', list('abcdefgh'))
    self.assertIn('=', result)

  def test_eqratio(self):
    result = pretty.pretty_nl('eqratio', list('abcdefgh'))
    self.assertIn(':', result)
    self.assertIn('=', result)

  def test_aconst(self):
    result = pretty.pretty_nl('aconst', ['a', 'b', 'c', 'd', '45'])
    self.assertIn('45', result)
    self.assertIn('=', result)

  def test_rconst(self):
    result = pretty.pretty_nl('rconst', ['a', 'b', 'c', 'd', '2'])
    self.assertIn('2', result)
    self.assertIn(':', result)

  def test_acompute(self):
    result = pretty.pretty_nl('acompute', ['a', 'b', 'c', 'd'])
    self.assertIsNotNone(result)

  def test_collx(self):
    result = pretty.pretty_nl('collx', ['a', 'b', 'c'])
    self.assertIn('collinear', result)

  def test_symbol_aliases(self):
    self.assertEqual(
        pretty.pretty_nl('C', ['a', 'b', 'c']),
        pretty.pretty_nl('coll', ['a', 'b', 'c']),
    )
    self.assertEqual(
        pretty.pretty_nl('O', ['a', 'b', 'c', 'd']),
        pretty.pretty_nl('cyclic', ['a', 'b', 'c', 'd']),
    )
    self.assertEqual(
        pretty.pretty_nl('M', ['x', 'a', 'b']),
        pretty.pretty_nl('midp', ['x', 'a', 'b']),
    )
    self.assertEqual(
        pretty.pretty_nl('D', ['a', 'b', 'c', 'd']),
        pretty.pretty_nl('cong', ['a', 'b', 'c', 'd']),
    )
    self.assertEqual(
        pretty.pretty_nl('T', ['a', 'b', 'c', 'd']),
        pretty.pretty_nl('perp', ['a', 'b', 'c', 'd']),
    )
    self.assertEqual(
        pretty.pretty_nl('P', ['a', 'b', 'c', 'd']),
        pretty.pretty_nl('para', ['a', 'b', 'c', 'd']),
    )
    self.assertEqual(
        pretty.pretty_nl('^', list('abcdefgh')),
        pretty.pretty_nl('eqangle', list('abcdefgh')),
    )
    self.assertEqual(
        pretty.pretty_nl('/', list('abcdefgh')),
        pretty.pretty_nl('eqratio', list('abcdefgh')),
    )


class PrettyTest(unittest.TestCase):

  def test_coll(self):
    self.assertEqual(pretty.pretty('coll a b c'), 'C a b c')

  def test_collx(self):
    self.assertEqual(pretty.pretty('collx a b c'), 'X a b c')

  def test_cyclic(self):
    self.assertEqual(pretty.pretty('cyclic a b c d'), 'O a b c d')

  def test_midp(self):
    self.assertEqual(pretty.pretty('midp x a b'), 'M x a b')

  def test_midpoint_alias(self):
    self.assertEqual(pretty.pretty('midpoint x a b'), 'M x a b')

  def test_cong(self):
    self.assertEqual(pretty.pretty('cong a b c d'), 'D a b c d')

  def test_perp_four_args(self):
    self.assertEqual(pretty.pretty('perp a b c d'), 'T a b c d')

  def test_perp_two_args_algebraic(self):
    self.assertEqual(pretty.pretty('perp ab cd'), 'T ab cd')

  def test_para_four_args(self):
    self.assertEqual(pretty.pretty('para a b c d'), 'P a b c d')

  def test_para_two_args_algebraic(self):
    self.assertEqual(pretty.pretty('para ab cd'), 'P ab cd')

  def test_eqangle(self):
    self.assertEqual(pretty.pretty('eqangle a b c d e f g h'), '^ a b c d e f g h')

  def test_eqratio(self):
    self.assertEqual(pretty.pretty('eqratio a b c d e f g h'), '/ a b c d e f g h')

  def test_eqratio3(self):
    self.assertEqual(pretty.pretty('eqratio3 a b c d o o'), 'S o a b o c d')

  def test_simtri(self):
    self.assertEqual(pretty.pretty('simtri a b c x y z'), 'S a b c x y z')

  def test_simtri2(self):
    self.assertEqual(pretty.pretty('simtri2 a b c x y z'), 'S a b c x y z')

  def test_contri(self):
    self.assertEqual(pretty.pretty('contri a b c x y z'), '= a b c x y z')

  def test_contri2(self):
    self.assertEqual(pretty.pretty('contri2 a b c x y z'), '= a b c x y z')

  def test_circle(self):
    self.assertEqual(pretty.pretty('circle o a b c'), 'I o a b c')

  def test_foot(self):
    self.assertEqual(pretty.pretty('foot a b c d'), 'F a b c d')

  def test_aconst(self):
    self.assertEqual(pretty.pretty('aconst a b c d 45'), '^ a b c d 45')

  def test_rconst(self):
    self.assertEqual(pretty.pretty('rconst a b c d 2'), '/ a b c d 2')

  def test_acompute(self):
    self.assertEqual(pretty.pretty('acompute a b c d'), 'A a b c d')

  def test_rcompute(self):
    self.assertEqual(pretty.pretty('rcompute a b c d'), 'R a b c d')

  def test_fixc(self):
    self.assertEqual(pretty.pretty('fixc a b c'), 'Q a b c')

  def test_fixl(self):
    self.assertEqual(pretty.pretty('fixl a b'), 'E a b')

  def test_fixb(self):
    self.assertEqual(pretty.pretty('fixb a b'), 'V a b')

  def test_fixt(self):
    self.assertEqual(pretty.pretty('fixt a b'), 'H a b')

  def test_fixp(self):
    self.assertEqual(pretty.pretty('fixp a b'), 'Z a b')

  def test_ind(self):
    self.assertEqual(pretty.pretty('ind a b c'), 'Y a b c')

  def test_list_input(self):
    # pretty() accepts a list as well as a string
    self.assertEqual(pretty.pretty(['coll', 'a', 'b', 'c']), 'C a b c')

  def test_unknown_predicate_passthrough(self):
    # Unknown predicates are returned as-is
    self.assertEqual(pretty.pretty('unknown x y z'), 'unknown x y z')


if __name__ == '__main__':
  absltest.main()
