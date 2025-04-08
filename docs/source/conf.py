"""Configuration file for the Sphinx documentation builder."""

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'EMLE-Bespoke'
copyright = '2024'
author = 'EMLE-Bespoke Team'
version = '0.1.0'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Add mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static'] 