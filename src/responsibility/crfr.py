"""Consensus responsibility factor routing (CRFR).

The implementation is extracted from the retained shared responsibility-gate
experiment.  It keeps the original equations, parameter names, initialization
order, and forward interface while using a normal package import.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import MASTER


GATE_MODES: Tuple[str, ...] = ("shared", "stock_specific")
STOCK_STATE_FIELDS: Tuple[str, ...] = (
    "c_1",
    "c_5",
    "c_20",
    "c_60",
    "e_1",
    "e_5",
    "e_20",
    "e_60",
    "beta",
    "R2",
    "residual_volatility",
    "abnormal_volume_20",
    "valid_ratio_60",
    "volume_valid",
)
RULE_COMPONENT_FIELDS: Tuple[str, ...] = (
    "c_1",
    "c_5",
    "c_20",
    "c_60",
    "e_1",
    "e_5",
    "e_20",
    "e_60",
)
SELECTION_LABELS: Tuple[str, ...] = ("off", "continuation", "reversal")
RULE_LABELS: Tuple[str, ...] = (
    "common_continuation",
    "common_reversal",
    "specific_continuation",
    "specific_reversal",
)


class MechanismResponsibilityFactorGateMASTER(MASTER):
    """Backbone with an economic-mechanism responsibility factor gate."""

    MARKET_DIM = 63
    STOCK_STATE_DIM = 14
    RULE_COMPONENT_DIM = 8
    DIRECTION_COUNT = 4
    FEATURE_DIM = 158
    MARKET_HIDDEN_DIM = 16
    SELECTOR_INPUT_DIM = 30
    SELECTOR_HIDDEN_DIM = 16
    SELECTOR_COUNT = 3
    DIRECTION_CLIP = 5.0
    PREFERENCE_EPS = 1e-6

    _NEW_MODULE_PREFIXES: Tuple[str, ...] = (
        "market_norm.",
        "market_projection.",
        "selector_norm.",
        "selector_hidden.",
        "common_selector.",
        "specific_selector.",
        "horizon_logits.",
        "mechanism_preference",
        "alpha_logit",
    )

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
        gate_mode: str,
        rule_seed: int,
        selection_temperature: float = 1.0,
        alpha_initial: float = 0.1,
    ) -> None:
        # Construct the backbone first so its same-seed initialization is unchanged.
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
        )
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if d_feat != self.FEATURE_DIM:
            raise ValueError(f"d_feat must be 158, got {d_feat}")
        if self.d_gate_input != self.MARKET_DIM:
            raise ValueError(
                f"Market gate input must have 63 features, got {self.d_gate_input}"
            )
        if not math.isfinite(selection_temperature) or selection_temperature <= 0:
            raise ValueError("selection_temperature must be finite and positive")
        if not 0.0 < float(alpha_initial) < 1.0:
            raise ValueError("alpha_initial must be strictly between zero and one")

        self.gate_mode = gate_mode
        self.rule_seed = int(rule_seed)
        self.selection_temperature = float(selection_temperature)
        reference = next(self.parameters())
        factory_kwargs = {"device": reference.device, "dtype": reference.dtype}
        fork_devices = []
        if reference.device.type == "cuda":
            index = reference.device.index
            fork_devices = [torch.cuda.current_device() if index is None else index]

        # Added initialization is paired across modes and does not advance the
        # caller's global Torch RNG sequence.
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.random.default_generator.manual_seed(self.rule_seed)
            if fork_devices:
                with torch.cuda.device(fork_devices[0]):
                    torch.cuda.manual_seed(self.rule_seed)

            self.market_norm = nn.LayerNorm(
                self.MARKET_DIM, eps=1e-5, **factory_kwargs
            )
            self.market_projection = nn.Linear(
                self.MARKET_DIM,
                self.MARKET_HIDDEN_DIM,
                bias=True,
                **factory_kwargs,
            )
            self.selector_norm = nn.LayerNorm(
                self.SELECTOR_INPUT_DIM, eps=1e-5, **factory_kwargs
            )
            self.selector_hidden = nn.Linear(
                self.SELECTOR_INPUT_DIM,
                self.SELECTOR_HIDDEN_DIM,
                bias=True,
                **factory_kwargs,
            )
            self.common_selector = nn.Linear(
                self.SELECTOR_HIDDEN_DIM,
                self.SELECTOR_COUNT,
                bias=True,
                **factory_kwargs,
            )
            self.specific_selector = nn.Linear(
                self.SELECTOR_HIDDEN_DIM,
                self.SELECTOR_COUNT,
                bias=True,
                **factory_kwargs,
            )
            prior_logits = reference.new_tensor((0.8, 0.1, 0.1)).log()
            for selector in (self.common_selector, self.specific_selector):
                nn.init.zeros_(selector.weight)
                with torch.no_grad():
                    selector.bias.copy_(prior_logits)

            self.horizon_logits = nn.ParameterDict(
                {
                    "rule_1": nn.Parameter(reference.new_zeros(3)),
                    "rule_2": nn.Parameter(reference.new_zeros(2)),
                    "rule_3": nn.Parameter(reference.new_zeros(3)),
                    "rule_4": nn.Parameter(reference.new_zeros(2)),
                }
            )
            self.mechanism_preference = nn.Parameter(
                torch.empty(
                    self.DIRECTION_COUNT,
                    self.FEATURE_DIM,
                    **factory_kwargs,
                )
            )
            nn.init.normal_(self.mechanism_preference, mean=0.0, std=1.0)
            alpha_tensor = reference.new_tensor(float(alpha_initial))
            self.alpha_logit = nn.Parameter(torch.logit(alpha_tensor).reshape(()))

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def horizon_weights(self) -> Dict[str, Tensor]:
        return {
            name: torch.softmax(logits, dim=0)
            for name, logits in self.horizon_logits.items()
        }

    def normalized_mechanism_preferences(self) -> Tensor:
        centered = self.mechanism_preference - self.mechanism_preference.mean(
            dim=1, keepdim=True
        )
        rms = torch.sqrt(centered.square().mean(dim=1, keepdim=True))
        return centered / rms.clamp_min(self.PREFERENCE_EPS)

    def new_module_parameter_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name in ("mechanism_preference", "alpha_logit")
            or name.startswith(self._NEW_MODULE_PREFIXES)
        )

    def new_module_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        names = set(self.new_module_parameter_names())
        for name, parameter in self.named_parameters():
            if name in names:
                yield name, parameter

    def new_module_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.new_module_named_parameters():
            yield parameter

    def master_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        names = set(self.new_module_parameter_names())
        for name, parameter in self.named_parameters():
            if name not in names:
                yield name, parameter

    def master_parameters(self) -> Iterator[nn.Parameter]:
        for _, parameter in self.master_named_parameters():
            yield parameter

    def compute_raw_directions(self, rule_components: Tensor) -> Tensor:
        self._validate_matrix(
            rule_components,
            name="rule_components",
            rows=rule_components.shape[0],
            columns=self.RULE_COMPONENT_DIM,
        )
        weights = self.horizon_weights()
        c_1, c_5, c_20, c_60, e_1, e_5, e_20, e_60 = rule_components.unbind(1)
        d_1 = (
            weights["rule_1"][0] * c_5
            + weights["rule_1"][1] * c_20
            + weights["rule_1"][2] * c_60
        )
        d_2 = -(weights["rule_2"][0] * c_1 + weights["rule_2"][1] * c_5)
        d_3 = (
            weights["rule_3"][0] * e_5
            + weights["rule_3"][1] * e_20
            + weights["rule_3"][2] * e_60
        )
        d_4 = -(weights["rule_4"][0] * e_1 + weights["rule_4"][1] * e_5)
        return torch.stack((d_1, d_2, d_3, d_4), dim=1)

    def normalize_directions(
        self,
        raw_directions: Tensor,
        direction_scale: Tensor,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        self._validate_vector(
            direction_scale, name="direction_scale", length=self.DIRECTION_COUNT
        )
        if not torch.isfinite(direction_scale).all().item():
            raise ValueError("direction_scale contains NaN or Inf")
        if not torch.all(direction_scale > 0).item():
            raise ValueError("direction_scale entries must be strictly positive")
        if valid_mask.ndim != 1 or valid_mask.shape[0] != raw_directions.shape[0]:
            raise ValueError("valid_mask must have shape [N]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        valid_float = valid_mask.to(raw_directions.dtype).unsqueeze(1)
        valid_count = valid_float.sum()
        valid_mean = (raw_directions * valid_float).sum(
            dim=0, keepdim=True
        ) / valid_count.clamp_min(1.0)
        centered = (raw_directions - valid_mean) * valid_float
        normalized = torch.clamp(
            centered / direction_scale.unsqueeze(0),
            min=-self.DIRECTION_CLIP,
            max=self.DIRECTION_CLIP,
        )
        return centered, normalized

    def compute_selection_probabilities(
        self,
        market_63: Tensor,
        stock_state: Tensor,
        *,
        selection_temperature: Union[float, None] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        temperature = (
            self.selection_temperature
            if selection_temperature is None
            else float(selection_temperature)
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("selection_temperature must be finite and positive")
        market_hidden = F.gelu(
            self.market_projection(self.market_norm(market_63)),
            approximate="none",
        )
        selector_input = torch.cat((market_hidden, stock_state), dim=1)
        selector_hidden = F.gelu(
            self.selector_hidden(self.selector_norm(selector_input)),
            approximate="none",
        )
        common_logits = self.common_selector(selector_hidden)
        specific_logits = self.specific_selector(selector_hidden)
        common_probabilities = torch.softmax(
            common_logits / temperature, dim=1
        )
        specific_probabilities = torch.softmax(
            specific_logits / temperature, dim=1
        )
        return (
            common_probabilities,
            specific_probabilities,
            common_logits,
            specific_logits,
        )

    def compute_gate_components(
        self,
        x: Tensor,
        stock_state: Tensor,
        rule_components: Tensor,
        direction_scale: Tensor,
        *,
        fusion_enabled: bool = True,
        selection_temperature: Union[float, None] = None,
    ) -> Dict[str, Tensor]:
        market_63 = x[
            :, -1, self.gate_input_start_index : self.gate_input_end_index
        ]
        market_logits = self.feature_gate.trans(market_63)
        raw_directions = self.compute_raw_directions(rule_components)
        rule_valid = rule_components.abs().sum(dim=1) > 0
        centered_directions, directions = self.normalize_directions(
            raw_directions, direction_scale, rule_valid
        )
        (
            common_probabilities,
            specific_probabilities,
            common_logits,
            specific_logits,
        ) = self.compute_selection_probabilities(
            market_63,
            stock_state,
            selection_temperature=selection_temperature,
        )
        selection_weights = torch.stack(
            (
                common_probabilities[:, 1],
                common_probabilities[:, 2],
                specific_probabilities[:, 1],
                specific_probabilities[:, 2],
            ),
            dim=1,
        )
        channel_coefficients = selection_weights * directions
        preferences = self.normalized_mechanism_preferences()
        channel_corrections = (
            self.alpha
            * channel_coefficients.unsqueeze(-1)
            * preferences.unsqueeze(0)
        )
        stock_correction = channel_corrections.sum(dim=1)

        if self.gate_mode == "shared":
            valid_float = rule_valid.to(stock_correction.dtype).unsqueeze(1)
            shared_correction = (stock_correction * valid_float).sum(
                dim=0, keepdim=True
            ) / valid_float.sum().clamp_min(1.0)
            effective_correction = shared_correction.expand_as(stock_correction)
        else:
            effective_correction = stock_correction

        applied_correction = (
            effective_correction
            if fusion_enabled
            else torch.zeros_like(effective_correction)
        )

        # beta applies to the complete market-plus-responsibility logits.
        gate_weights = self.feature_gate.d_output * torch.softmax(
            (market_logits + applied_correction) / self.feature_gate.t,
            dim=-1,
        )
        common_conflict = common_probabilities[:, 1] * common_probabilities[:, 2]
        specific_conflict = (
            specific_probabilities[:, 1] * specific_probabilities[:, 2]
        )
        return {
            "market_63": market_63,
            "market_logits": market_logits,
            "raw_directions": raw_directions,
            "rule_valid": rule_valid,
            "centered_directions": centered_directions,
            "directions": directions,
            "common_probabilities": common_probabilities,
            "specific_probabilities": specific_probabilities,
            "common_logits": common_logits,
            "specific_logits": specific_logits,
            "selection_probabilities": torch.stack(
                (common_probabilities, specific_probabilities), dim=1
            ),
            "selection_weights": selection_weights,
            "channel_coefficients": channel_coefficients,
            "mechanism_preferences": preferences,
            "channel_logit_corrections": channel_corrections,
            "stock_logit_correction": stock_correction,
            "effective_logit_correction": effective_correction,
            "applied_logit_correction": applied_correction,
            "gate_weights": gate_weights,
            "alpha": self.alpha,
            "common_conflict": common_conflict,
            "specific_conflict": specific_conflict,
            "conflict_loss": (common_conflict + specific_conflict).mean(),
        }

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
            raise ValueError(f"x must have shape [N,T,F], got {list(x.shape)}")
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
            direction_scale, name="direction_scale", length=self.DIRECTION_COUNT
        )
        for name, value in (
            ("stock_state", stock_state),
            ("rule_components", rule_components),
            ("direction_scale", direction_scale),
        ):
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError(f"{name} must match x device and dtype")

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
        prediction = self.layers(src).squeeze(-1)
        if not return_aux:
            return prediction
        auxiliary: Dict[str, Any] = dict(components)
        auxiliary["direction_scale"] = direction_scale
        auxiliary["horizon_weights"] = self.horizon_weights()
        auxiliary["gate_mode"] = self.gate_mode
        auxiliary["fusion_enabled"] = bool(fusion_enabled)
        return prediction, auxiliary

    @staticmethod
    def _validate_matrix(
        value: Tensor, *, name: str, rows: int, columns: int
    ) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape != (rows, columns):
            raise ValueError(
                f"{name} must have shape [{rows},{columns}], got {list(value.shape)}"
            )

    @staticmethod
    def _validate_vector(value: Tensor, *, name: str, length: int) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1 or value.shape[0] != length:
            raise ValueError(f"{name} must have shape [{length}], got {list(value.shape)}")


MechanismResponsibilityGateMASTER = MechanismResponsibilityFactorGateMASTER
DeepRuleFusionMASTER = MechanismResponsibilityFactorGateMASTER


__all__ = [
    "GATE_MODES",
    "STOCK_STATE_FIELDS",
    "RULE_COMPONENT_FIELDS",
    "SELECTION_LABELS",
    "RULE_LABELS",
    "MechanismResponsibilityFactorGateMASTER",
    "MechanismResponsibilityGateMASTER",
    "DeepRuleFusionMASTER",
]
