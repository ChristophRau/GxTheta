
from distutils.core import setup

setup(name='pyLMM-GxTheta',
      version='0.2',
      author="Christoph Rau",
      author_email='chrau@ucla.edu',
      url="https://github.com/ChristophRau/GxTheta",
      description='pyLMM is a lightweight linear mixed model solver for use in GWAS from Nick Furlotte.  GxTheta expands the analysis to look for sites with variable effect size due to ancestry',
      packages=['pylmmGxT'],
      scripts=['scripts/pylmmGxTheta.py', 'scripts/pylmmKinship.py',  'scripts/estimate_variance_components.py'],
      )
