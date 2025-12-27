# Changes Needed to Implement TPM-Style Training (on top of TRM-style code)

This checklist assumes you already have a TRM-like training loop that:
- runs iterative refinement for `T` steps,
- produces intermediate predictions `y_hat[t]`,
- applies deep supervision (loss at each step),
- typically detaches state between steps to keep memory bounded.

Goal: modify training so recursion behaves like *refinement*:
1) discourage stepwise regressions (progress objective),
2) allow short-horizon cross-step credit assignment (two-step gradient bridge),
while keeping inference unchanged.

References:
- Tiny Progressive Model (TPM) paper fileciteturn0file0
- Less is More / Tiny Recursive Model (TRM) paper fileciteturn1file0

---

## 0) What should NOT change
- **Inference**: keep the same iterative loop; just run more steps for better results.
- **Model core**: keep your operator `Fθ(x, s_t) -> s_{t+1}` and decoder `decode(s_t) -> y_hat_t`.
- **Deep supervision** can stay (TPM keeps it), but its gradients need the bridge.

---

## 1) Add Progress Objective (Stepwise Loss Decrease)

### 1.1 Compute per-step task potential `V_t`
Define `V_t` as your standard supervised loss at step `t`:
- token/grid classification: `V_t = CE(y_hat[t], y_true)`
- use your existing differentiable task loss

### 1.2 Add progress penalty `L_prog`
Penalize stepwise regressions:
- for `t = 0..T-2`:
  - `delta = V[t+1] - V[t] + margin_m`
  - `penalty = phi(delta)` where `phi` is `relu` (hinge) or `softplus`
- `L_prog = sum(penalty)`

Recommended defaults:
- `phi = softplus` (smoother)
- `margin_m = 0.0` (or small positive)
- `lambda_prog` tuned on validation

### 1.3 Total loss
Keep deep supervision:
- `L_sup = sum_t w_t * V_t` (uniform or mildly increasing weights)
Total:
- `L = L_sup + lambda_prog * L_prog`
(Optionally add halting loss if you already have one.)

---

## 2) Add Two-Step Gradient Bridge (Adjacent-step credit assignment)

### 2.1 The issue to fix
If you do `state = stopgrad(F(x, state))` between every step, then the loss at step `t+1`
cannot improve how the step-`t` state was produced.

### 2.2 Target behavior
Allow gradients to flow across *two adjacent steps* (`K=2`) without full unroll/BPTT.

### 2.3 Implementation pattern (rollout + segment replay)

#### Phase A — Detached rollout (on-policy trajectory)
Generate a length-`T` trajectory **without storing activations**:
- `s[0] = init_state(x)`
- for `t in 0..T-1`: `s[t+1] = F(x, s[t])` under `torch.no_grad()` (or detach)
Store `s[t]` tensors (detached) for segment sampling.

#### Phase B — Sample short segments and backprop through 2 steps
For each optimizer step, sample segment start indices `t ∈ [0, T-2]` and re-run with grads:

```
s_t  = s_rollout[t]          # detached
s_t1 = F(x, s_t)             # grad ON
V1   = loss(decode(s_t1), y_true)

s_t2 = F(x, s_t1)            # grad ON
V2   = loss(decode(s_t2), y_true)

Lsup_seg  = w[t+1]*V1 + w[t+2]*V2
Lprog_seg = phi((V2 - V1) + margin_m)

L_seg = Lsup_seg + lambda_prog * Lprog_seg
```

Key point: gradients from `V2` flow into the computation that produced `s_t1`.

### 2.4 How to combine with full-step deep supervision
Recommended (closest to TPM):
- compute the full rollout under no-grad to get on-policy states
- train using **only** short 2-step segments (possibly multiple per batch/example)

---

## 3) Code changes (practical)

### 3.1 Refactor forward for segment replay
Expose:
- `step(x, state) -> next_state`
- `decode(state) -> y_hat`
This makes segment replay clean.

### 3.2 Store rollout states
During Phase A, store:
- `states = [s0, s1, ..., sT]` (detached)

If memory is tight:
- store every k steps or store in fp16/bf16
- sample fewer segments

### 3.3 Segment sampler
Implement:
- uniform sampling over `[0, T-2]`, or
- biased-to-late sampling (often helps late-step behavior)

---

## 4) Hyperparameters to add

- `T`: refinement steps
- `lambda_prog`: weight for progress loss
- `margin_m`: progress margin
- `phi_type`: `hinge` or `softplus`
- `K`: fixed to 2
- `num_segments_per_batch`: how many segments to replay
- `w_t`: step weights (uniform or mildly increasing)

---

## 5) Metrics to add (to verify it works)

Evaluate anytime behavior:
- **progress curve**: mean `V_t` vs step
- **regression rate**: fraction with `V_{t+1} > V_t`
- **compute–accuracy**: accuracy at step budgets {1,2,4,8,...,T}
- optionally: best-so-far vs last-step accuracy

---

## 6) Minimal PyTorch-style pseudocode

### 6.1 Rollout (detached)
```
with torch.no_grad():
    s = init_state(x)
    states = [s]
    for t in range(T):
        s = step(x, s)
        states.append(s)
```

### 6.2 Segment training (K=2)
```
loss_total = 0
for _ in range(num_segments):
    t = sample_t(0, T-2)
    s_t = states[t].detach()

    s_t1 = step(x, s_t)
    V1 = task_loss(decode(s_t1), y_true)

    s_t2 = step(x, s_t1)
    V2 = task_loss(decode(s_t2), y_true)

    Lsup = w[t+1]*V1 + w[t+2]*V2
    Lprog = phi((V2 - V1) + margin_m)

    loss_total += Lsup + lambda_prog * Lprog

loss_total.backward()
opt.step()
opt.zero_grad()
```

---

## 7) Implementation task list for a coding agent

1. Add per-step potential computation `V_t`.
2. Implement `L_prog` with margin + hinge/softplus.
3. Implement detached rollout to collect `states`.
4. Implement 2-step segment replay with gradients.
5. Train with `L_sup + lambda_prog * L_prog` (at least within segments).
6. Add evaluation metrics: progress curves + regression rate + compute–accuracy.
7. Expose new knobs in config and log them.
