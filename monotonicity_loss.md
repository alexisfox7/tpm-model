# Monotonicity / Progress Loss (TPM change #1)

This repo currently refines predictions across **outer ACT steps** (each call to `model(carry=..., batch=...)` advances `carry.steps` and produces a new set of logits).

The “monotonicity loss” described in `tpm_over_trm_changes.md` can be implemented as a **progress objective** over consecutive ACT steps:

- Define per-step potential `V_t` = mean task loss (CE) for the *current* forward.
- Penalize regressions: `V_t > V_{t-1}` (optionally with a margin).

This doc shows the minimal code edits (as snippets) to add:

- `L_prog = sum(phi((V_t - V_{t-1}) + margin_m))`
- `L_total = L_existing + lambda_prog * L_prog`

## Summary of files to change

- `models/recursive_reasoning/trm.py`
  - Add `prev_task_loss` to the ACT carry.
- `models/losses.py`
  - Extend `ACTLossHead` to compute and apply the progress loss.
- `config/arch/trm.yaml`
  - Add config knobs for `lambda_prog`, `margin_m`, `phi_type`.

If you train other architectures (HRM / baseline transformer), you should apply the same carry change there as well.

---

## 1) Add `prev_task_loss` to the ACT carry

### 1.1 Modify the carry dataclass (TRM)

Edit: `models/recursive_reasoning/trm.py`

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]

    # NEW: previous step's per-example task loss (mean CE over valid tokens)
    prev_task_loss: torch.Tensor
```

### 1.2 Initialize it in `initial_carry`

Edit: `models/recursive_reasoning/trm.py`

```python
def initial_carry(self, batch: Dict[str, torch.Tensor]):
    batch_size = batch["inputs"].shape[0]

    return TinyRecursiveReasoningModel_ACTV1Carry(
        inner_carry=self.inner.empty_carry(batch_size),
        steps=torch.zeros((batch_size, ), dtype=torch.int32),
        halted=torch.ones((batch_size, ), dtype=torch.bool),
        current_data={k: torch.empty_like(v) for k, v in batch.items()},

        # NEW
        prev_task_loss=torch.zeros((batch_size,), dtype=torch.float32, device=batch["inputs"].device),
    )
```

### 1.3 Ensure `forward()` passes it through

Edit: `models/recursive_reasoning/trm.py`

At the end of `forward()`, include `carry.prev_task_loss` in the returned carry (it will be updated by the loss head; see below):

```python
return TinyRecursiveReasoningModel_ACTV1Carry(
    new_inner_carry,
    new_steps,
    halted,
    new_current_data,
    prev_task_loss=carry.prev_task_loss,
), outputs
```

Notes:

- The carry field is used by `ACTLossHead` to compute `V_t - V_{t-1}`.
- You *must* avoid comparing losses across different puzzles. We’ll handle that by masking with `was_reset = incoming_carry.halted` in the loss head.

---

## 2) Add the progress loss inside `ACTLossHead`

Edit: `models/losses.py`

### 2.1 Add hyperparameters to the loss head init

```python
class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, lambda_prog: float = 0.0, margin_m: float = 0.0, phi_type: str = "softplus"):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

        # NEW
        self.lambda_prog = float(lambda_prog)
        self.margin_m = float(margin_m)
        self.phi_type = str(phi_type)
```

### 2.2 Compute per-example potential `V_t`

After you compute `mask`, `loss_counts`, `loss_divisor`, compute a **per-example** mean CE:

```python
loss_per_token = self.loss_fn(
    outputs["logits"],
    labels,
    ignore_index=IGNORE_LABEL_ID,
    valid_mask=mask,
)  # shape: (B, L)

V_t = loss_per_token.sum(-1) / loss_counts.clamp_min(1)  # shape: (B,)
```

### 2.3 Compute the progress penalty vs `prev_task_loss`

Use the *incoming* carry (the argument passed into the loss head) to get:

- which examples were reset this step (`incoming_carry.halted == True` means “new puzzle loaded”)
- the previous per-example loss

```python
incoming_carry = model_kwargs["carry"]
prev_V = incoming_carry.prev_task_loss.to(V_t.dtype)

# Do not compare across puzzles:
# If an example was reset, we don't have a meaningful previous V.
was_reset = incoming_carry.halted
valid_prog = (~was_reset) & (loss_counts > 0)

delta = (V_t - prev_V) + self.margin_m

if self.phi_type == "hinge":
    penalty = F.relu(delta)
elif self.phi_type == "softplus":
    penalty = F.softplus(delta)
else:
    raise ValueError(f"Unknown phi_type: {self.phi_type}")

L_prog = torch.where(valid_prog, penalty, torch.zeros_like(penalty)).sum()
```

### 2.4 Add it to total loss and update carry

Replace the existing LM loss computation so you can reuse `loss_per_token`:

```python
lm_loss = (loss_per_token / loss_divisor.unsqueeze(-1)).sum()

loss_total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

if self.lambda_prog != 0.0:
    loss_total = loss_total + self.lambda_prog * L_prog
```

Then update the new carry before returning:

```python
new_carry.prev_task_loss = V_t.detach().to(torch.float32)
```

(If your carry dataclass is immutable in practice, return a new carry instance with the updated field.)

### 2.5 Add metrics

Add (summed scalars) so `pretrain.py` reduction works unchanged:

```python
with torch.no_grad():
    metrics.update({
        "prog_loss": L_prog.detach(),
        "regression_count": torch.where(valid_prog & (delta > 0), 1, 0).sum(),
        "prog_count": valid_prog.sum(),
    })
```

In W&B you can interpret:

- `train/regression_rate ≈ regression_count / prog_count`

(You can compute the ratio on the logging side if desired.)

---

## 3) Add config knobs under the TRM arch loss

Edit: `config/arch/trm.yaml`

```yaml
loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy

  # NEW (defaults are safe/no-op)
  lambda_prog: 0.0
  margin_m: 0.0
  phi_type: softplus
```

Setting `lambda_prog: 0.0` keeps behavior identical to current training.

---

## 4) Notes / gotchas

- **What “step” means here**
  - This implements progress across **outer ACT steps** (consecutive calls to the model).
  - It does *not* enforce progress across inner `H_cycles/L_cycles` inside `TinyRecursiveReasoningModel_ACTV1_Inner.forward()`.

- **Don’t compare across puzzles**
  - A batch slot can switch to a new puzzle when it halts.
  - Mask using `incoming_carry.halted` so you only compare within the same puzzle trajectory.

- **Scale**
  - `L_prog` is a sum over batch positions (like the other losses in this repo).
  - Tune `lambda_prog` accordingly (often small, e.g. `1e-2` to `1e-1`, but depends on your CE scale).

---

## 5) Optional extension: apply carry change to other arch files

If you train other ACT models, repeat the carry field addition in:

- `models/recursive_reasoning/hrm.py` (`HierarchicalReasoningModel_ACTV1Carry`)
- `models/recursive_reasoning/transformers_baseline.py` (`Model_ACTV2Carry`)
