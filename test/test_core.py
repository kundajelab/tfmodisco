from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from modisco import core
import time
from nose.tools import raises


class TestSnippet(unittest.TestCase):

    def setUp(self):
        pass
        
    def test_snippet_basic(self): 
        fwd = [1,2,3,4] 
        rev = [-1,-2,-3,-4]
        snippet = core.Snippet(fwd=fwd, rev=rev, has_pos_axis=False)
        rev_snippet = snippet.revcomp() 
        self.assertEqual(len(fwd), len(snippet))
        self.assertListEqual(snippet.fwd, rev_snippet.rev) 
        self.assertListEqual(snippet.rev, rev_snippet.fwd)

    @raises(AssertionError)
    def test_snippet_length_tracks_error(self): 
        fwd = [1,2,3,4,5] 
        rev = [-1,-2,-3,-4]
        snippet = core.Snippet(fwd=fwd, rev=rev, has_pos_axis=False)
