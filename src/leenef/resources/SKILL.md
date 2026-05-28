# leenef method-selection skill

Use this skill when choosing a leenef method:

- static supervised data -> `NEFLayer`;
- image data -> receptive-field encoders or `ConvNEFPipeline`;
- sequence classification -> `StreamingNEFClassifier`;
- continual task streams -> `partial_fit` plus `solve_accumulated`;
- dense discrete-action RL -> `NEFFQIAgent`;
- sparse-reward RL research -> `NEFSFAgent`.

Prefer accumulated `A^T A` / `A^T Y` workflows for online, continual, or large
GPU runs.  Prefer better encoders or front ends before simply increasing neuron
count when accuracy is feature-limited.  For RL, treat target construction as
the main design problem: stable non-bootstrapped targets for dense control,
goal/successor/hindsight targets for sparse control.

The installed helper mirrors this decision process:

```bash
leenef-recommend supervised --input-kind image
leenef-recommend temporal --input-kind sequence --gpu
leenef-recommend continual --input-kind image
leenef-recommend rl --gpu
```
