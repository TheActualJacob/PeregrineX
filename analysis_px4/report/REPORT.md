# PX4 neural controller vs classical stack — adversarial campaign

Runs: 800 attempted, 800 scored (100%). Settled at setpoint before scoring: 800.

## Headline

| metric | classical | neural |
|---|---|---|
| failure rate | 24.5% | 27.0% |
| crashes | 92 | 108 |
| median lateral drift, crash scenarios | 0.10 m | 1.51 m |

Paired on identical scenarios, the neural controller scores a lower margin in 47% of 400 scenarios (mean difference -0.040).

## Reading it

Both controllers lose altitude under a large enough mass step, so the pass/fail boundary is similar. The difference is in HOW they fail: in crash scenarios the network's lateral drift median is 14.7x the classical stack's — it loses position control as well as altitude, where PID holds lateral position while descending.

## Caveats

- SIH (simple internal hover model), not Gazebo: aerodynamics are coarse and PX4's `failure motor off` injection is NOT honored by this backend (verified inert), so motor_fail rows are a null lever kept visible on purpose.
- The policy shipped in mainline is trained for an X500 V2 frame; the SIH quad is not that airframe, so some degradation is expected and the classical arm is the control for exactly that reason.
- Hover/position-hold only. No trajectory tracking, no wind field.
- Oracle clamps any crash to margin -1.0, so severity beyond the crash threshold lives in `max_dev_m`, not in the margin.
