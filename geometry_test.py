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

"""Unit tests for geometry.py."""
import unittest

from absl.testing import absltest
import geometry as gm


class GeometryTest(unittest.TestCase):

  def _setup_equality_example(self):
    # Create 4 nodes a, b, c, d
    # and their lengths
    a = gm.Segment('a')
    la = gm.Length('l(a)')
    a.connect_to(la)
    la.connect_to(a)

    b = gm.Segment('b')
    lb = gm.Length('l(b)')
    b.connect_to(lb)
    lb.connect_to(b)

    c = gm.Segment('c')
    lc = gm.Length('l(c)')
    c.connect_to(lc)
    lc.connect_to(c)

    d = gm.Segment('d')
    ld = gm.Length('l(d)')
    d.connect_to(ld)
    ld.connect_to(d)

    # Now let a=b, b=c, a=c, c=d
    la.merge([lb], 'fact1')
    lb.merge([lc], 'fact2')
    la.merge([lc], 'fact3')
    lc.merge([ld], 'fact4')
    return a, b, c, d, la, lb, lc, ld

  def test_merged_node_representative(self):
    _, _, _, _, la, lb, lc, ld = self._setup_equality_example()

    # all nodes are now represented by la.
    self.assertEqual(la.rep(), la)
    self.assertEqual(lb.rep(), la)
    self.assertEqual(lc.rep(), la)
    self.assertEqual(ld.rep(), la)

  def test_merged_node_equivalence(self):
    _, _, _, _, la, lb, lc, ld = self._setup_equality_example()
    # all la, lb, lc, ld are equivalent
    self.assertCountEqual(la.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(lb.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(lc.equivs(), [la, lb, lc, ld])
    self.assertCountEqual(ld.equivs(), [la, lb, lc, ld])

  def test_bfs_for_equality_transitivity(self):
    a, _, _, d, _, _, _, _ = self._setup_equality_example()

    # check that a==d because fact3 & fact4, not fact1 & fact2
    self.assertCountEqual(gm.why_equal(a, d), ['fact3', 'fact4'])

  def test_neighbors(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    l.connect_to(c)
    self.assertCountEqual(l.neighbors(gm.Point), [a, b, c])

  def test_merge_edge_graph(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    l1 = gm.Line('l1')
    l2 = gm.Line('l2')
    l1.connect_to(a)
    l1.connect_to(b)
    l2.connect_to(c)
    l2.connect_to(d)
    l1.merge_edge_graph(l2.edge_graph)
    self.assertCountEqual(l1.neighbors(gm.Point), [a, b, c, d])

  def test_connect_to(self):
    a = gm.Point('a')
    b = gm.Point('b')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    self.assertCountEqual(l.neighbors(gm.Point), [a, b])

  def test_equivs_upto(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    a.merge([b], 'fact1')
    b.merge([c], 'fact2')
    c.merge([d], 'fact3')
    self.assertCountEqual(a.equivs_upto(2).keys(), [a, b, c])

  def test_why_equal(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    a.merge([b], 'fact1')
    b.merge([c], 'fact2')
    c.merge([d], 'fact3')
    self.assertCountEqual(a.why_equal([d], 2), ['fact1', 'fact2'])

  def test_why_equal_groups(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    e = gm.Point('e')
    f = gm.Point('f')
    a.merge([b], 'fact1')
    b.merge([c], 'fact2')
    c.merge([d], 'fact3')
    d.merge([e], 'fact4')
    e.merge([f], 'fact5')
    groups = [[d, e], [f]]
    self.assertCountEqual(a.why_equal_groups(groups, 3)[0], ['fact1', 'fact2', 'fact3'])

  def test_why_val(self):
    a = gm.Segment('a')
    la = gm.Length('l(a)')
    a.connect_to(la)
    la.connect_to(a)
    self.assertEqual(a.why_val(1), [])

  def test_why_connect(self):
    a = gm.Point('a')
    b = gm.Point('b')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    self.assertEqual(a.why_connect(b), [None])

  def test_is_equiv(self):
    a = gm.Point('a')
    b = gm.Point('b')
    a.merge([b], 'fact1')
    self.assertTrue(gm.is_equiv(a, b))

  def test_is_equal(self):
    a = gm.Segment('a')
    la = gm.Length('l(a)')
    a.connect_to(la)
    la.connect_to(a)
    b = gm.Segment('b')
    lb = gm.Length('l(b)')
    b.connect_to(lb)
    lb.connect_to(b)
    la.merge([lb], 'fact1')
    self.assertTrue(gm.is_equal(a, b))

  def test_bfs_backtrack(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    a.merge([b], 'fact1')
    b.merge([c], 'fact2')
    c.merge([d], 'fact3')
    parent = {a: None, b: a, c: b, d: c}
    self.assertCountEqual(gm.bfs_backtrack(a, [d], parent), ['fact1', 'fact2', 'fact3'])

  def test_why_coll(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    l.connect_to(c)
    self.assertEqual(l.why_coll([a, b, c]), [None, None, None])

  def test_why_cyclic(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    circ = gm.Circle('circ')
    circ.connect_to(a)
    circ.connect_to(b)
    circ.connect_to(c)
    circ.connect_to(d)
    self.assertEqual(circ.why_cyclic([a, b, c, d]), [None, None, None, None])

  def test_why_equal(self):
    a = gm.Point('a')
    b = gm.Point('b')
    a.merge([b], 'fact1')
    self.assertEqual(gm.why_equal(a, b), [])

  def test_get_lines_thru_all(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    l.connect_to(c)
    self.assertEqual(gm.get_lines_thru_all(a, b, c), [l])

  def test_line_of_and_why(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    l = gm.Line('l')
    l.connect_to(a)
    l.connect_to(b)
    l.connect_to(c)
    self.assertEqual(gm.line_of_and_why([a, b, c]), (l, [None, None, None]))

  def test_get_circles_thru_all(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    circ = gm.Circle('circ')
    circ.connect_to(a)
    circ.connect_to(b)
    circ.connect_to(c)
    circ.connect_to(d)
    self.assertEqual(gm.get_circles_thru_all(a, b, c, d), [circ])

  def test_circle_of_and_why(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    circ = gm.Circle('circ')
    circ.connect_to(a)
    circ.connect_to(b)
    circ.connect_to(c)
    circ.connect_to(d)
    self.assertEqual(gm.circle_of_and_why([a, b, c, d]), (circ, [None, None, None, None]))

  def test_name_map(self):
    a = gm.Point('a')
    b = gm.Point('b')
    c = gm.Point('c')
    d = gm.Point('d')
    self.assertEqual(gm.name_map([a, b, c, d]), ['a', 'b', 'c', 'd'])

  def test_new_val(self):
    l = gm.Line('l')
    self.assertIsInstance(l.new_val(), gm.Direction)

  def test_set_directions(self):
    a = gm.Direction('a')
    b = gm.Direction('b')
    ang = gm.Angle('ang')
    ang.set_directions(a, b)
    self.assertEqual(ang.directions, (a, b))

  def test_set_lengths(self):
    a = gm.Length('a')
    b = gm.Length('b')
    r = gm.Ratio('r')
    r.set_lengths(a, b)
    self.assertEqual(r.lengths, (a, b))

  def test_all_angles(self):
    a = gm.Direction('a')
    b = gm.Direction('b')
    ang = gm.Angle('ang')
    ang.set_directions(a, b)
    self.assertEqual(list(gm.all_angles(a, b)), [(ang, {a: None}, {b: None})])

  def test_all_ratios(self):
    a = gm.Length('a')
    b = gm.Length('b')
    r = gm.Ratio('r')
    r.set_lengths(a, b)
    self.assertEqual(list(gm.all_ratios(a, b)), [(r, {a: None}, {b: None})])

  def test_val_type(self):
    l = gm.Line('l')
    self.assertEqual(gm.val_type(l), gm.Direction)
