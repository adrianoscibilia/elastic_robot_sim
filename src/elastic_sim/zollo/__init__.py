"""Model-based elastic-joint baseline: Zollo et al. (2015) identification.

The identification is the linear-in-parameters elastic-joint model of L.
Zollo, E. Lopez, L. Spedaliere, N. Garcia Aracil, E. Guglielmelli,
"Identification of Dynamic Parameters for Robots with Elastic Joints",
Adv. Mech. Eng. 7(2), 2015: link rows and motor rows stacked into
``T = W Pi``, the independent columns found by elimination, the base vector
solved by least squares.  It runs on the simulated training split only; the
predictors built from it (:mod:`.predictors`) are then scored zero-shot like
the learned models (`REFACTOR_SPECS/model_based_baseline/01_zollo_plan.md`).

Modules:

* :mod:`.regressor` -- the column layout and the per-sample regressor rows;
* :mod:`.basis` -- incremental QR, base parameters, categorization, solve;
* :mod:`.signals` -- Zollo's preprocessing (filter, differentiate, parallel
  filter, decimate, low-speed rejection);
* :mod:`.data` -- loading the producer's files, delay alignment, ground truth;
* :mod:`.identify` -- the configuration and the identification pipeline;
* :mod:`.predictors` -- ID-inverse, ID-rigid, ID-elastic and the scoring rows;
* :mod:`.synthetic` -- a self-contained Spong-model simulator for the
  recovery checks (gate G1).
"""
