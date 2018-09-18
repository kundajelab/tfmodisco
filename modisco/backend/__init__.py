from __future__ import division, absolute_import, print_function
import os
import json
import sys

#This code is based on the keras backend code

# Set TF-MoDISco base dir path given TFMODISCO_HOME env variable, if applicable.
# Otherwise either ~/.tfmodisco or /tmp.
if 'TFMODISCO_HOME' in os.environ:
    _tfmodisco_dir = os.environ.get('TFMODISCO_HOME')
else:
    _tfmodisco_base_dir = os.path.expanduser('~')
    if not os.access(_tfmodisco_base_dir, os.W_OK):
        _tfmodisco_base_dir = '/tmp'
    _tfmodisco_dir = os.path.join(_tfmodisco_base_dir, '.tfmodisco')

# Default backend: TensorFlow.
_BACKEND = 'tensorflow'

# Attempt to read tfmodisco config file.
_config_path = os.path.expanduser(os.path.join(_tfmodisco_dir, 'tfmodisco.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _backend = _config.get('backend', _BACKEND)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_tfmodisco_dir):
    try:
        os.makedirs(_tfmodisco_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'backend': _BACKEND,
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on TFMODISCO_BACKEND flag, if applicable.
if 'TFMODISCO_BACKEND' in os.environ:
    _backend = os.environ['TFMODISCO_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'theano':
    sys.stderr.write('TF-MoDISco is using the Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('TF-MoDISco is using the TensorFlow backend.\n')
    from .tensorflow_backend import *
else:
    raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend tfmodisco is currently using.
    # Example
    ```python
        >>> tfmodisco.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
