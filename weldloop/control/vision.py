"""RGB-vision control arm — included to be measured failing.

This exists because "why not just use a camera?" is the first question anyone
asks, and an assertion is a weaker answer than a number.

It is deliberately built as the **most favourable possible** version of the
idea, so that what it demonstrates is a property of the sensor rather than of
a weak implementation:

* identical motion-layer control law to the adaptive controller;
* identical physics-prior EKF, with the same process model and the same
  covariance handling;
* the camera is even allowed to see the pool width directly, which a real
  vision pipeline would have to infer.

The only difference is which measurements reach the filter: the RGB pool-width
reading instead of the power-source ripple, the profiler and the IR camera.
Whatever gap opens up between this arm and the adaptive one is therefore
attributable to the sensor, not the controller.

This is emphatically **not** a VLA or a learned visual policy, and the demo
does not claim it is.  A learned policy on this input would inherit the same
problem: during arc-on, roughly three quarters of the frames are unusable
(Phase 2), and the ones that survive carry a pool-width error comparable to
the pool itself.  No policy class fixes a blinded sensor.
"""

from __future__ import annotations

from weldloop.config import WeldConfig
from weldloop.control.motion_layer import AdaptiveController

__all__ = ["VisionController"]


class VisionController(AdaptiveController):
    """Same control law as :class:`AdaptiveController`, RGB-only state estimate."""

    def __init__(self, cfg: WeldConfig, **kw) -> None:
        kw.pop("sensor_set", None)
        kw.setdefault("name", "vision")
        super().__init__(cfg, sensor_set="rgb", **kw)
