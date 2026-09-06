"""Stock-specific responsibility-routed cross-sectional aggregation (RRCA).

The retained path is::

    shared CRFR -> temporal attention -> cross-sectional attention
    -> stock-specific RRCA -> temporal pooling -> prediction head

The historical experiment called this block ``MechanismGroupedContext``.
Both historical class names and state-dict keys are retained for checkpoint
compatibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .backbone import SAttention, TAttention
from .crfr import MechanismResponsibilityFactorGateMASTER


class MechanismGroupedContext(nn.Module):
    """Four sum-to-one mechanism contexts with stock-specific reception."""

    def __init__(
        self,
        d_model: int = 256,
        mechanism_count: int = 4,
        adapter_rank: int = 32,
        rho_initial: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or mechanism_count <= 0 or adapter_rank <= 0:
            raise ValueError("module dimensions must be positive")
        if not math.isfinite(float(rho_initial)) or not 0.0 < rho_initial < 1.0:
            raise ValueError("rho_initial must be finite and strictly between 0 and 1")

        self.d_model = int(d_model)
        self.mechanism_count = int(mechanism_count)
        self.adapter_rank = int(adapter_rank)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model, eps=1e-5),
                    nn.Linear(self.d_model, self.adapter_rank),
                    nn.GELU(),
                    nn.Linear(self.adapter_rank, self.d_model),
                )
                for _ in range(self.mechanism_count)
            ]
        )

        # A small, non-zero output initialization lets the responsibility
        # network receive a useful gradient immediately while rho keeps the
        # initial perturbation close to CRFR.
        for adapter in self.adapters:
            output = adapter[-1]
            nn.init.normal_(output.weight, mean=0.0, std=0.01)
            nn.init.zeros_(output.bias)

        rho_tensor = torch.tensor(float(rho_initial))
        self.rho_logit = nn.Parameter(torch.logit(rho_tensor).reshape(()))

    @property
    def rho(self) -> Tensor:
        return torch.sigmoid(self.rho_logit)

    def forward(
        self,
        hidden: Tensor,
        responsibility: Tensor,
        valid_mask: Tensor,
        *,
        enabled: bool = True,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [N,T,D]")
        stocks, _, channels = hidden.shape
        if channels != self.d_model:
            raise ValueError(
                "hidden channel mismatch: expected %d, got %d"
                % (self.d_model, channels)
            )
        expected = (stocks, self.mechanism_count)
        if responsibility.shape != expected:
            raise ValueError(
                "responsibility must have shape %s, got %s"
                % (list(expected), list(responsibility.shape))
            )
        if valid_mask.shape != (stocks,) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [N]")
        if responsibility.device != hidden.device or responsibility.dtype != hidden.dtype:
            raise ValueError("responsibility must match hidden device and dtype")
        if valid_mask.device != hidden.device:
            raise ValueError("valid_mask must match hidden device")

        valid = valid_mask.to(dtype=hidden.dtype).unsqueeze(1)
        source_mass = responsibility * valid
        source_denominator = source_mass.sum(dim=0, keepdim=True)
        source_weights = source_mass / source_denominator.clamp_min(
            torch.finfo(hidden.dtype).eps
        )

        # [N,K] x [N,T,D] -> [K,T,D]. Each mechanism has its own context.
        contexts = torch.einsum("nk,ntd->ktd", source_weights, hidden)
        adapted_contexts = torch.stack(
            [adapter(contexts[k]) for k, adapter in enumerate(self.adapters)],
            dim=0,
        )

        target_responsibility = responsibility * valid
        delta = torch.einsum(
            "nk,ktd->ntd", target_responsibility, adapted_contexts
        )
        if not enabled:
            delta = torch.zeros_like(delta)
        output = hidden + self.rho * delta

        if not return_aux:
            return output
        return output, {
            "source_weights": source_weights,
            "source_weight_sums": source_weights.sum(dim=0),
            "contexts": contexts,
            "adapted_contexts": adapted_contexts,
            "target_responsibility": target_responsibility,
            "delta": delta,
            "rho": self.rho,
        }


class MechanismGroupedContextMASTER(MechanismResponsibilityFactorGateMASTER):
    """Shared CRFR gate with four stock-specific responsibility contexts."""

    MECHANISM_COUNT = 4
    DEFAULT_ADAPTER_RANK = 32

    def __init__(
        self,
        d_feat: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        gate_input_start_index: int,
        gate_input_end_index: int,
        beta: float,
        *,
        rule_seed: int,
        selection_temperature: float = 1.0,
        alpha_initial: float = 0.1,
        rho_initial: float = 0.1,
        adapter_rank: int = DEFAULT_ADAPTER_RANK,
        context_seed: Optional[int] = None,
        gate_mode: str = "shared",
    ) -> None:
        if gate_mode != "shared":
            raise ValueError("MGC retains the result-exact G1 shared factor gate")
        if d_model != 256:
            raise ValueError("MGC currently fixes d_model=256")

        super().__init__(
            d_feat=d_feat,
            d_model=d_model,
            t_nhead=t_nhead,
            s_nhead=s_nhead,
            T_dropout_rate=T_dropout_rate,
            S_dropout_rate=S_dropout_rate,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            beta=beta,
            gate_mode="shared",
            rule_seed=rule_seed,
            selection_temperature=selection_temperature,
            alpha_initial=alpha_initial,
        )

        self.context_seed = int(
            int(rule_seed) + 0x4D4743 if context_seed is None else context_seed
        )
        reference = next(self.parameters())
        fork_devices = []
        if reference.device.type == "cuda":
            device_index = reference.device.index
            fork_devices = [
                torch.cuda.current_device()
                if device_index is None
                else int(device_index)
            ]
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.random.default_generator.manual_seed(self.context_seed)
            if fork_devices:
                with torch.cuda.device(fork_devices[0]):
                    torch.cuda.manual_seed(self.context_seed)
            self.mechanism_context = MechanismGroupedContext(
                d_model=d_model,
                mechanism_count=self.MECHANISM_COUNT,
                adapter_rank=int(adapter_rank),
                rho_initial=rho_initial,
            ).to(device=reference.device, dtype=reference.dtype)

    def g1_module_parameter_names(self) -> Tuple[str, ...]:
        """Responsibility-factor-gate parameter names for diagnostics."""

        return tuple(super().new_module_parameter_names())

    def context_body_parameter_names(self) -> Tuple[str, ...]:
        """Low-rank context adapter parameter names for diagnostics."""

        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.startswith("mechanism_context.adapters.")
        )

    def condition_parameter_names(self) -> Tuple[str, ...]:
        """Residual-strength parameter name for diagnostics."""

        return tuple(
            name
            for name, _ in self.named_parameters()
            if name == "mechanism_context.rho_logit"
        )

    def context_module_parameter_names(self) -> Tuple[str, ...]:
        """All grouped-context parameters, exposed for diagnostics."""

        return self.context_body_parameter_names() + self.condition_parameter_names()

    def new_module_parameter_names(self) -> Tuple[str, ...]:
        return self.g1_module_parameter_names() + self.context_module_parameter_names()

    def g1_module_parameters(self) -> Iterator[nn.Parameter]:
        names = set(self.g1_module_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield parameter

    def context_module_parameters(self) -> Iterator[nn.Parameter]:
        names = set(self.context_module_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield parameter

    def context_body_parameters(self) -> Iterator[nn.Parameter]:
        names = set(self.context_body_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield parameter

    def condition_parameters(self) -> Iterator[nn.Parameter]:
        names = set(self.condition_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield parameter

    def new_module_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        names = set(self.new_module_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield name, parameter

    def new_module_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.new_module_named_parameters():
            yield parameter

    def forward(
        self,
        x: Tensor,
        stock_state: Tensor,
        rule_components: Tensor,
        direction_scale: Tensor,
        *,
        return_aux: bool = False,
        fusion_enabled: bool = True,
        selection_temperature: Union[float, None] = None,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        if x.ndim != 3:
            raise ValueError("x must have shape [N,T,F], got %s" % list(x.shape))
        rows = x.shape[0]
        if x.shape[2] < self.gate_input_end_index:
            raise ValueError("x feature dimension is smaller than gate end index")
        self._validate_matrix(
            stock_state,
            name="stock_state",
            rows=rows,
            columns=self.STOCK_STATE_DIM,
        )
        self._validate_matrix(
            rule_components,
            name="rule_components",
            rows=rows,
            columns=self.RULE_COMPONENT_DIM,
        )
        self._validate_vector(
            direction_scale,
            name="direction_scale",
            length=self.DIRECTION_COUNT,
        )
        for name, value in (
            ("stock_state", stock_state),
            ("rule_components", rule_components),
            ("direction_scale", direction_scale),
        ):
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError("%s must match x device and dtype" % name)

        components = self.compute_gate_components(
            x,
            stock_state,
            rule_components,
            direction_scale,
            fusion_enabled=fusion_enabled,
            selection_temperature=selection_temperature,
        )
        src = x[:, :, : self.gate_input_start_index]
        src = src * components["gate_weights"].unsqueeze(1)

        hidden = self.layers[0](src)
        hidden = self.layers[1](hidden)
        hidden = self.layers[2](hidden)
        hidden = self.layers[3](hidden)
        hidden, context_aux = self.mechanism_context(
            hidden,
            components["selection_weights"],
            components["rule_valid"],
            enabled=fusion_enabled,
            return_aux=True,
        )
        pooled = self.layers[4](hidden)
        prediction = self.layers[5](pooled).squeeze(-1)
        if not return_aux:
            return prediction

        auxiliary: Dict[str, Any] = dict(components)
        auxiliary.update(
            {
                "direction_scale": direction_scale,
                "horizon_weights": self.horizon_weights(),
                "gate_mode": self.gate_mode,
                "fusion_enabled": bool(fusion_enabled),
                "context_responsibility": components["selection_weights"],
                "context_source_weights": context_aux["source_weights"],
                "context_source_weight_sums": context_aux[
                    "source_weight_sums"
                ],
                "mechanism_contexts": context_aux["contexts"],
                "adapted_mechanism_contexts": context_aux[
                    "adapted_contexts"
                ],
                "context_target_responsibility": context_aux[
                    "target_responsibility"
                ],
                "context_delta": context_aux["delta"],
                "context_rho": context_aux["rho"],
                "context_output": hidden,
            }
        )
        return prediction, auxiliary


MGCMASTER = MechanismGroupedContextMASTER
RRCAMASTER = MechanismGroupedContextMASTER


__all__ = [
    "TAttention",
    "SAttention",
    "MechanismGroupedContext",
    "MechanismGroupedContextMASTER",
    "MGCMASTER",
    "RRCAMASTER",
]
