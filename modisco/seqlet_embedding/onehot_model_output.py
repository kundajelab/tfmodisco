from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from .. import core as modiscocore


class ModelOutputEmbedderFactory(object):

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self, onehot_track_name,
                 toscore_track_names_and_signs):
        return self.cls(*self.args,
                        onehot_track_name=onehot_track_name,
                        **self.kwargs) 

    def get_jsonable_config(self):
        return OrderedDict([('cls', self.cls.__name__),
                            ('args', self.args), ('kwargs', self.kwargs)]) 


class ModelOutputEmbedder(object):

    def __init__(self, prediction_func, onehot_track_name):
        self.prediction_func = prediction_func 
        self.onehot_track_name = onehot_track_name

    def __call__(self, seqlets):
        
        onehot_track_fwd, onehot_track_rev =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=[self.onehot_track_name],
                track_transformer=None)
        
        embedding_fwd = self.prediction_func(onehot_track_fwd) 
        if (onehot_track_rev is not None):
            embedding_rev = self.prediction_func(onehot_track_rev) 
        else:
            embedding_rev = None

        return embedding_fwd, embedding_rev

    @classmethod
    def get_factory(cls, *args, **kwargs):
        return ModelOutputEmbedderFactory(cls=cls, *args, **kwargs) 


class KerasModelOutputEmbedder(ModelOutputEmbedder):

    def __init__(self, model_h5, **kwargs):
        import keras
        from keras.models import load_model 
        self.keras_model = load_model(model_h5)
        super(KerasModelOutputEmbedder, self).__init__(
              prediction_func=self.keras_model.predict, **kwargs)
