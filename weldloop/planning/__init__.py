"""Seconds-to-minutes task planning (deliberately outside the real-time loop).

``task_planner`` defines what a plan is and gives the rule-based floor;
``vla`` does the same job from a job card and a fit-up image with a
vision-language model; ``schedule`` turns either one into a table the 20 ms
layer reads with a lookup, after clamping every setpoint the cell will not
accept.
"""
