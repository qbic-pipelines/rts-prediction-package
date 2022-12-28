================================
Root-Tissue-Segmentation Package
================================

.. image:: https://github.com/qbic-pipelines/rts-prediction-package/workflows/Build%20rts_package%20Package/badge.svg
        :target: https://github.com/qbic-pipelines/rts-prediction-package/workflows/Build%20rts_package%20Package/badge.svg
        :alt: Github Workflow Build rts_package Status

.. image:: https://github.com/qbic-pipelines/rts-prediction-package/workflows/Run%20rts_package%20Tox%20Test%20Suite/badge.svg
        :target: https://github.com/qbic-pipelines/rts-prediction-package/workflows/Run%20rts_package%20Tox%20Test%20Suite/badge.svg
        :alt: Github Workflow Tests Status

.. image:: https://img.shields.io/pypi/v/rts_package.svg
        :target: https://pypi.python.org/pypi/rts_package
        :alt: PyPI Status


.. image:: https://readthedocs.org/projects/rts_package/badge/?version=latest
        :target: https://rts_package.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://flat.badgen.net/dependabot/thepracticaldev/dev.to?icon=dependabot
        :target: https://flat.badgen.net/dependabot/thepracticaldev/dev.to?icon=dependabot
        :alt: Dependabot Enabled


Prediction package for reproducible U-Net models, trained for semantic segmentation of microscopy images of root tissue from *A. thaliana* (https://github.com/qbic-pipelines/root-tissue-segmentation-core/). These models are trained using the mlf-core framework and tested for reproducibility. This package can be deployed within an analysis pipeline as a module for root tissue segmentation (rts) of fluorescence microscopy images.

* This package can be installed via pip: https://pypi.org/project/root-tissue-seg-package/

* The trained pytorch model used for prediction can be found here: https://zenodo.org/record/6937290/

Package Tools
-------------

* Segmentation prediction CLI: ``rts-pred``
* Uncertainty of prediction CLI: ``rts-pred-uncert``
* Input feature importance (Guided Grad-CAM) CLI: ``rts-feat-imp``

Usage Examples
--------------

* ``rts-pred -i ./brightfields -o ./predictions -m mark1-PHDFM-u2net-model.ckpt --suffix ""``
* ``rts-pred-uncert -i ./brightfields -o ./predictions -m mark1-PHDFM-u2net-model.ckpt --suffix "" -t 5``
* ``rts-feat-imp -i ./brightfields -o ./predictions -m mark1-PHDFM-u2net-model.ckpt --suffix "" -t 2``

Additional Information
-------------

* Free software under MIT license

* Documentation: https://rts-package.readthedocs.io.

Credits
-------

This package was created with mlf-core_ using cookiecutter_.


.. _mlf-core: https://mlf-core.com
.. _cookiecutter: https://github.com/audreyr/cookiecutter
