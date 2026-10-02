"""Unit tests for the KnowledgeBase (kb.py)."""

from __future__ import annotations

import unittest

from kb import Fact, KnowledgeBase


class TestFact(unittest.TestCase):
    def test_str_no_args(self):
        self.assertEqual(str(Fact("foo")), "foo")

    def test_str_with_args(self):
        self.assertEqual(str(Fact("at", ("a", "b"))), "at(a, b)")

    def test_equality_and_hash(self):
        self.assertEqual(Fact("at", ("a", "b")), Fact("at", ("a", "b")))
        self.assertNotEqual(Fact("at", ("a", "b")), Fact("at", ("b", "a")))
        s = {Fact("at", ("a", "b")), Fact("at", ("a", "b"))}
        self.assertEqual(len(s), 1)


class TestKnowledgeBase(unittest.TestCase):
    def test_add_and_contains(self):
        kb = KnowledgeBase()
        self.assertTrue(kb.add(Fact("at", ("a", "b"))))
        self.assertTrue(kb.contains(Fact("at", ("a", "b"))))
        self.assertFalse(kb.contains(Fact("at", ("a", "c"))))

    def test_add_duplicate_returns_false(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        self.assertFalse(kb.add(Fact("at", ("a", "b"))))

    def test_remove(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        self.assertTrue(kb.remove(Fact("at", ("a", "b"))))
        self.assertFalse(kb.contains(Fact("at", ("a", "b"))))
        self.assertFalse(kb.remove(Fact("at", ("a", "b"))))

    def test_tuple_form_accepted(self):
        kb = KnowledgeBase()
        kb.add(("at", ("a", "b")))
        self.assertTrue(kb.contains(("at", ("a", "b"))))

    def test_query_wildcard(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        kb.add(Fact("at", ("c", "d")))
        kb.add(Fact("road", ("a", "b")))
        matches = kb.query("at", None, None)
        self.assertEqual(len(matches), 2)

    def test_query_first_arg_bound(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        kb.add(Fact("at", ("c", "d")))
        matches = kb.query("at", "a", None)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].args, ("a", "b"))

    def test_query_ground(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        self.assertEqual(len(kb.query("at", "a", "b")), 1)
        self.assertEqual(len(kb.query("at", "a", "c")), 0)

    def test_has_predicate(self):
        kb = KnowledgeBase()
        self.assertFalse(kb.has_predicate("at"))
        kb.add(Fact("at", ("a", "b")))
        self.assertTrue(kb.has_predicate("at"))

    def test_remove_matching(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        kb.add(Fact("at", ("a", "c")))
        kb.add(Fact("at", ("d", "e")))
        removed = kb.remove_matching("at", "a", None)
        self.assertEqual(removed, 2)
        self.assertEqual(len(kb.query("at", None, None)), 1)

    def test_clear(self):
        kb = KnowledgeBase()
        kb.add(Fact("at", ("a", "b")))
        kb.clear()
        self.assertEqual(len(kb), 0)
        self.assertEqual(len(kb.facts()), 0)

    def test_from_facts(self):
        facts = [Fact("a", ("x",)), Fact("b", ("y",))]
        kb = KnowledgeBase.from_facts(facts)
        self.assertTrue(kb.contains(facts[0]))
        self.assertTrue(kb.contains(facts[1]))


if __name__ == "__main__":
    unittest.main()