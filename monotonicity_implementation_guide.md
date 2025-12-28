# Monotonicity Loss Implementation Guide

This guide explains the current code structure and the minimal changes needed to add monotonicity loss.

## Overview

The monotonicity loss penalizes the model when the task loss **increases** across consecutive ACT steps, encouraging monotonic improvement in predictions.

**Key idea**: Track `V_t` (current step's mean CE loss per example) and penalize when `V_t > V_{t-1}`.

---

## File 1: `models/recursive_reasoning/trm.py`

### Current Code: Carry Dataclass

The carry stores state between ACT steps:

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]
```

**What it does**:
- `inner_carry`: Hidden states (z_H, z_L)
- `steps`: How many ACT steps each example has taken
- `halted`: Whether each example finished (loaded new puzzle)
- `current_data`: Current batch data for each slot

### Change #1: Add prev_task_loss field

```python
@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]

    # NEW: Track previous step's per-example task loss
    prev_task_loss: torch.Tensor
```

---

### Current Code: Initialize Carry

```python
def initial_carry(self, batch: Dict[str, torch.Tensor]):
    batch_size = batch["inputs"].shape[0]

    return TinyRecursiveReasoningModel_ACTV1Carry(
        inner_carry=self.inner.empty_carry(batch_size),

        steps=torch.zeros((batch_size, ), dtype=torch.int32),
        halted=torch.ones((batch_size, ), dtype=torch.bool),

        current_data={k: torch.empty_like(v) for k, v in batch.items()}
    )
```

**What it does**: Creates initial carry state. All examples start as "halted" (will load new puzzles).

### Change #2: Initialize prev_task_loss to zero

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

---

### Current Code: Forward Return

```python
return TinyRecursiveReasoningModel_ACTV1Carry(
    new_inner_carry,
    new_steps,
    halted,
    new_current_data
), outputs
```

**What it does**: Returns updated carry and model outputs (logits, Q-values).

### Change #3: Pass through prev_task_loss

```python
return TinyRecursiveReasoningModel_ACTV1Carry(
    new_inner_carry,
    new_steps,
    halted,
    new_current_data,
    prev_task_loss=carry.prev_task_loss,  # NEW: pass through (will be updated in loss head)
), outputs
```

**Why**: The loss head will update `prev_task_loss` with the current step's loss.

---

## File 2: `models/losses.py`

### Current Code: ACTLossHead Init

```python
class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
```

**What it does**: Wraps the model with loss computation. Uses `stablemax_cross_entropy` or `softmax_cross_entropy`.

### Change #4: Add monotonicity hyperparameters

```python
class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str,
                 lambda_prog: float = 0.0, margin_m: float = 0.0, phi_type: str = "softplus"):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

        # NEW: Monotonicity loss config
        self.lambda_prog = float(lambda_prog)
        self.margin_m = float(margin_m)
        self.phi_type = str(phi_type)
```

**Parameters**:
- `lambda_prog`: Weight for progress loss (0.0 = disabled)
- `margin_m`: Tolerance margin (allow small regressions)
- `phi_type`: Penalty function ("softplus" or "hinge")

---

### Current Code: Loss Computation

```python
# Around line 67-88
with torch.no_grad():
    # Preds
    outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

    # Correctness
    mask = (labels != IGNORE_LABEL_ID)
    loss_counts = mask.sum(-1)
    loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

    is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
    seq_is_correct = is_correct.sum(-1) == loss_counts

    # Metrics (halted)
    valid_metrics = new_carry.halted & (loss_counts > 0)
    metrics = {
        "count": valid_metrics.sum(),
        "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
        "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
        "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
        "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
    }

# Losses
lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
```

**What it does**:
1. Computes which tokens are valid (`mask`)
2. Computes LM loss (cross-entropy, averaged over valid tokens, summed over batch)
3. Computes Q-learning loss for halt prediction

---

### Change #5: Add monotonicity loss computation

Insert after the existing loss computation (around line 88-92):

```python
# Losses
# Compute per-token loss first
loss_per_token = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)  # (B, L)

# LM loss (existing)
lm_loss = (loss_per_token / loss_divisor).sum()

# Q-halt loss (existing)
q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

# NEW: Monotonicity / Progress Loss
L_prog = torch.tensor(0.0, device=lm_loss.device)
if self.lambda_prog != 0.0:
    # Per-example potential (mean CE loss)
    V_t = loss_per_token.sum(-1) / loss_counts.clamp_min(1)  # (B,)

    # Get previous step's loss from incoming carry
    incoming_carry = model_kwargs["carry"]
    prev_V = incoming_carry.prev_task_loss.to(V_t.dtype)

    # Don't compare across puzzles (halted=True means new puzzle loaded)
    # Also require valid tokens
    valid_prog = (~incoming_carry.halted) & (loss_counts > 0)

    # Compute regression delta
    delta = (V_t - prev_V) + self.margin_m

    # Apply penalty function
    if self.phi_type == "hinge":
        penalty = F.relu(delta)
    elif self.phi_type == "softplus":
        penalty = F.softplus(delta)
    else:
        raise ValueError(f"Unknown phi_type: {self.phi_type}")

    # Sum penalty over valid positions
    L_prog = torch.where(valid_prog, penalty, torch.zeros_like(penalty)).sum()

    # Update carry with current V_t for next step
    new_carry.prev_task_loss = V_t.detach().to(torch.float32)

    # Add metrics
    with torch.no_grad():
        metrics.update({
            "prog_loss": L_prog.detach(),
            "regression_count": torch.where(valid_prog & (delta > 0), torch.ones_like(delta), torch.zeros_like(delta)).sum(),
            "prog_count": valid_prog.sum(),
        })

metrics.update({
    "lm_loss": lm_loss.detach(),
    "q_halt_loss": q_halt_loss.detach(),
})
```

---

### Current Code: Final Loss Return

```python
# Q continue (bootstrapping target loss)
q_continue_loss = 0
if "target_q_continue" in outputs:
    q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
    metrics["q_continue_loss"] = q_continue_loss.detach()

# Filter outputs for return
detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
```

### Change #6: Add progress loss to total

```python
# Q continue (bootstrapping target loss)
q_continue_loss = 0
if "target_q_continue" in outputs:
    q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
    metrics["q_continue_loss"] = q_continue_loss.detach()

# Filter outputs for return
detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

# NEW: Add progress loss to total
total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
if self.lambda_prog != 0.0:
    total_loss = total_loss + self.lambda_prog * L_prog

return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
```

---

## File 3: `config/arch/trm.yaml`

### Current Code

```yaml
name: recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1
loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy

halt_exploration_prob: 0.1
halt_max_steps: 16
```

### Change #7: Add monotonicity config knobs

```yaml
name: recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1
loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy

  # NEW: Monotonicity loss (disabled by default)
  lambda_prog: 0.0      # Weight for progress loss (try 0.01-0.1 to enable)
  margin_m: 0.0         # Margin tolerance for regressions
  phi_type: softplus    # Penalty function: "softplus" or "hinge"

halt_exploration_prob: 0.1
halt_max_steps: 16
```

---

## Summary of Changes

| File | Lines | What to Add |
|------|-------|-------------|
| `trm.py` | ~30 | Add `prev_task_loss` field to carry |
| `trm.py` | ~247 | Initialize `prev_task_loss=torch.zeros(...)` |
| `trm.py` | ~297 | Pass through `prev_task_loss=carry.prev_task_loss` |
| `losses.py` | ~42 | Add `lambda_prog`, `margin_m`, `phi_type` params |
| `losses.py` | ~87-102 | Compute `V_t`, progress loss, update carry |
| `trm.yaml` | ~4 | Add config knobs under `loss:` |

---

## How to Use

1. **Default (disabled)**: Keep `lambda_prog: 0.0` - no behavior change
2. **Enable**: Set `lambda_prog: 0.05` - penalize regressions
3. **Tune margin**: Set `margin_m: 0.1` - allow small regressions without penalty
4. **Change penalty**: Set `phi_type: hinge` - use ReLU instead of softplus

**Monitor in W&B**:
- `train/prog_loss` - monotonicity penalty value
- `train/regression_count / train/prog_count` - fraction of steps that regressed

---

## Key Implementation Details

### Why track `prev_task_loss` in carry?
The carry persists across ACT steps. We need `V_{t-1}` to compute `V_t - V_{t-1}`.

### Why mask with `incoming_carry.halted`?
When `halted=True`, the batch slot loaded a new puzzle. Comparing losses across different puzzles is meaningless.

### Why detach `V_t` before storing?
We don't want gradients flowing through the previous step's loss into future steps.

### Why is `L_prog` summed over batch?
All losses in this codebase sum over batch (not mean), so reduction happens in `pretrain.py`.

---

## Technical Details & Shapes

### Function Signatures

#### `ACTLossHead.forward()`

```python
def forward(
    self,
    return_keys: Sequence[str],
    **model_kwargs,  # <-- Contains "carry" and "batch"
) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
    new_carry, outputs = self.model(**model_kwargs)
    # ...
```

**Key point**: `model_kwargs["carry"]` is the **incoming** carry (state from previous ACT step, before forward pass).

#### Loss functions return per-token losses

Both loss functions return `(B, L)` shaped tensors:

```python
def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # ...
    return -torch.where(valid_mask, prediction_logprobs, 0)  # (B, L)

def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # ...
    return F.cross_entropy(..., reduction="none").view(labels.shape)  # (B, L)
```

---

### Shape Flow Diagram

Here's how tensors flow through the monotonicity loss computation:

```
loss_per_token                           # (B, L) - CE loss per token
    ↓ .sum(-1)
V_t = loss_per_token.sum(-1)            # (B,) - sum over sequence
    ↓ / loss_counts.clamp_min(1)
V_t = ... / loss_counts.clamp_min(1)    # (B,) - mean over valid tokens

prev_V = incoming_carry.prev_task_loss  # (B,) - from previous step

delta = (V_t - prev_V) + margin_m       # (B,) - regression amount
    ↓ softplus/relu
penalty = F.softplus(delta)             # (B,) - penalty per example
    ↓ masked sum
L_prog = penalty[valid_prog].sum()      # scalar - total penalty
```

---

### Two Carries: Incoming vs Outgoing

This is critical for correctness:

```python
def forward(self, return_keys, **model_kwargs):
    # INCOMING carry: state from PREVIOUS step
    incoming_carry = model_kwargs["carry"]  # Has prev_task_loss from step t-1

    # Run model forward
    new_carry, outputs = self.model(**model_kwargs)

    # Compute V_t (current step's loss)
    V_t = loss_per_token.sum(-1) / loss_counts.clamp_min(1)

    # Compare with INCOMING carry's prev_task_loss
    prev_V = incoming_carry.prev_task_loss.to(V_t.dtype)
    delta = (V_t - prev_V) + self.margin_m

    # Update OUTGOING carry for NEXT step
    new_carry.prev_task_loss = V_t.detach().to(torch.float32)

    return new_carry, total_loss, ...
```

**Timeline**:
- Step t-1: Model stores `V_{t-1}` in outgoing carry
- Step t:
  - Read `V_{t-1}` from incoming carry
  - Compute `V_t`
  - Compare `V_t - V_{t-1}`
  - Store `V_t` in outgoing carry for step t+1

---

### Why `loss_counts.clamp_min(1)` instead of `loss_divisor`?

**Context**: Current code has:
```python
loss_counts = mask.sum(-1)                          # (B,) - num valid tokens
loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # (B, 1) - for broadcasting
```

**For LM loss** (operates on `(B, L)` tensors):
```python
lm_loss = (loss_per_token / loss_divisor).sum()
#          (B, L)          / (B, 1)      -> (B, L) -> scalar
```
Uses `loss_divisor` with shape `(B, 1)` to broadcast over sequence dimension.

**For V_t** (operates on `(B,)` tensors):
```python
V_t = loss_per_token.sum(-1) / loss_counts.clamp_min(1)
#     (B,)                    / (B,)
```
Uses `loss_counts.clamp_min(1)` with shape `(B,)` directly - no broadcasting needed.

**Wrong version** (would cause shape error):
```python
V_t = loss_per_token.sum(-1) / loss_divisor  # (B,) / (B, 1) -> broadcasts to (B, 1) ❌
```

**Correct version**:
```python
V_t = loss_per_token.sum(-1) / loss_counts.clamp_min(1)  # (B,) / (B,) -> (B,) ✅
```

---

### Element-wise Operations Safety

All monotonicity loss operations work on `(B,)` shaped tensors:

```python
V_t: (B,)                              # Current mean CE per example
prev_V: (B,)                           # Previous mean CE per example
delta: (B,)                            # V_t - prev_V + margin
penalty: (B,)                          # softplus(delta) or relu(delta)
valid_prog: (B,) bool                  # mask for valid comparisons
L_prog: scalar                         # penalty.sum() over valid examples
```

This ensures:
- No accidental broadcasting errors
- Clear per-example semantics
- Easy to mask with `valid_prog`
