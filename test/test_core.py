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

    def test_snippet_basic(self): 
        fwd = np.array([1,2,3,4])
        rev = np.array([-1,-2,-3,-4])
        snippet = core.Snippet(fwd=fwd, rev=rev, has_pos_axis=False)
        rev_snippet = snippet.revcomp() 
        self.assertEqual(len(fwd), len(snippet))
        np.testing.assert_almost_equal(snippet.fwd, rev_snippet.rev)
        np.testing.assert_almost_equal(snippet.rev, rev_snippet.fwd)

    @raises(AssertionError)
    def test_snippet_length_tracks_error(self): 
        fwd = [1,2,3,4,5] 
        rev = [-1,-2,-3,-4][::-1]
        snippet = core.Snippet(fwd=fwd, rev=rev, has_pos_axis=False)


class TestSeqletCoordinates(unittest.TestCase):

    def test_seqlet_coordinates_core(self):
        seqlet_coordinates = core.SeqletCoordinates(
                                example_idx=0, start=0, end=10,
                                is_revcomp=False)
        self.assertEqual(len(seqlet_coordinates), 10)


class TestDataTrack(unittest.TestCase):

    def setUp(self):
        self.fwd_tracks = np.arange(90).reshape((10,9))
        self.data_track_has_pos_axis = core.DataTrack( 
            name="track_has_pos_axis",
            fwd_tracks=self.fwd_tracks,
            rev_tracks=-self.fwd_tracks[:,::-1], has_pos_axis=True)
        self.data_track_no_pos_axis = core.DataTrack(  
            name="track_has_no_pos_axis",
            fwd_tracks=self.fwd_tracks,
            rev_tracks=-self.fwd_tracks[:,::-1], has_pos_axis=False)

    def test_data_track_basic(self):
        assert len(self.data_track_has_pos_axis)==10

    def test_data_track_get_snippet(self):
        np.testing.assert_almost_equal(
            self.data_track_has_pos_axis.get_snippet(core.SeqletCoordinates(
                                     example_idx=1, 
                                     start=1, end=5, is_revcomp=False)).fwd,
            self.fwd_tracks[1, 1:5])
        #snippet with pos axis
        np.testing.assert_almost_equal(
            self.data_track_has_pos_axis.get_snippet(core.SeqletCoordinates(
                                     example_idx=1, 
                                     start=1, end=5, is_revcomp=True)).fwd,
            -self.fwd_tracks[1, 1:5][::-1])
        #snippet without pos axis
        np.testing.assert_almost_equal(
            self.data_track_no_pos_axis.get_snippet(core.SeqletCoordinates(
                                     example_idx=1, 
                                     start=1, end=5, is_revcomp=False)).fwd,
            self.fwd_tracks[1])
        np.testing.assert_almost_equal(
            self.data_track_no_pos_axis.get_snippet(core.SeqletCoordinates(
                                     example_idx=1, 
                                     start=1, end=5, is_revcomp=True)).fwd,
            -self.fwd_tracks[1][::-1])
         

class TestSeqlet(unittest.TestCase):

    def setUp(self):
        self.fwd_tracks = np.arange(90).reshape((10,9))
        self.data_track_has_pos_axis = core.DataTrack( 
            name="track_has_pos_axis",
            fwd_tracks=self.fwd_tracks,
            rev_tracks=-self.fwd_tracks[:,::-1], has_pos_axis=True)
        self.data_track_no_pos_axis = core.DataTrack(  
            name="track_no_pos_axis",
            fwd_tracks=self.fwd_tracks,
            rev_tracks=-self.fwd_tracks[:,::-1], has_pos_axis=False)

    def test_seqlet_add_snippet_from_data_track_fwd(self):
        coor = core.SeqletCoordinates(example_idx=1, start=1,
                                      end=5, is_revcomp=False)  
        seqlet = core.Seqlet(coor=coor) 
        seqlet.add_snippet_from_data_track(
                data_track=self.data_track_has_pos_axis)
        seqlet.add_snippet_from_data_track(
                data_track=self.data_track_no_pos_axis)
        np.testing.assert_almost_equal(
            seqlet["track_has_pos_axis"].fwd, self.fwd_tracks[1, 1:5])
        np.testing.assert_almost_equal(
            seqlet["track_no_pos_axis"].fwd, self.fwd_tracks[1])

    def test_seqlet_add_snippet_from_data_track_rev(self):
        coor = core.SeqletCoordinates(example_idx=1, start=1,
                                      end=5, is_revcomp=True)  
        seqlet = core.Seqlet(coor=coor) 
        seqlet.add_snippet_from_data_track(
                data_track=self.data_track_has_pos_axis)
        seqlet.add_snippet_from_data_track(
                data_track=self.data_track_no_pos_axis)
        np.testing.assert_almost_equal(
            seqlet["track_has_pos_axis"].fwd, -self.fwd_tracks[1, 1:5][::-1])
        np.testing.assert_almost_equal(
            seqlet["track_no_pos_axis"].fwd, -self.fwd_tracks[1][::-1])


