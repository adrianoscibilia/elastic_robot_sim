"""ROS 2 real-data recording stack for the iiwa and UR10 (RR_01).

Pure-Python planning, config, conversion, identification and validation live
here with no ROS dependency; :mod:`erd_recording.pipeline` is the only module
that imports ``rclpy``, so ``config``/``planning``/``contract``/``convert``/
``identify``/``validate`` are importable and testable from plain ``pytest``
(T1.2, T1.5, T1.7).
"""
