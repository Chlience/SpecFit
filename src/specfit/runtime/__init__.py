
from specfit.trace import RuntimeEventKind

from .cache_observer import (
    CacheTransition,
    CacheTransitionCounters,
    CacheTransitionObserver,
    RecordingCacheTransitionObserver,
    aggregate_cache_transition_counters,
)
from .cache_snapshot import (
    CacheSlotSnapshot,
    GpuExpertCacheSnapshot,
    PendingTransferSnapshot,
)
from .eagle3_chain import (
    AuxHiddenStateCollector,
    EAGLE3_DEFAULT_TREE_DEPTH,
    EAGLE3_DEFAULT_TREE_NODE_BUDGET,
    EAGLE3_DEFAULT_TREE_TOP_K,
    EAGLE3_DEPTH5_MAX_DRAFTER_KV_TOKENS,
    EAGLE3_DEPTH5_MAX_FED_TOKENS,
    EAGLE3_MAX_COMMITTED_TOKENS,
    EAGLE3_VALIDATED_TOTAL_TOKENS,
    EAGLE3_VALIDATED_TREE_DEPTH,
    Eagle3ChainDrafter,
    GreedyChainSpeculator,
    QWEN3_EAGLE3_LAYER_INDICES,
    target_greedy_generate,
)
from .eagle3_equivalence import (
    LayerRouteMismatch,
    LayerRoutingEquivalence,
    NodeZeroNumericalReport,
    NodeVerificationEquivalence,
    TreeVerificationEquivalence,
    compare_tree_verification_to_causal_paths,
    diagnose_tree_node_zero_numerics,
)
from .eagle3_selection import (
    ExplicitRawIdTreeSelector,
    RawIdSelectionPolicy,
    TreeSelectionDecision,
    candidate_pool_fingerprint,
)
from .eagle3_trace import GreedyChainTraceRecorder
from .eagle3_tree import (
    CumulativeProbabilityTreeSelector,
    Eagle3TreeDrafter,
    PoolHindsightTreeSelector,
    ShallowTreeSpeculator,
)
from .eagle3_warmup import (
    eagle3_tree_temporary_kv_tokens,
    eagle3_tree_warmup_anchor_lengths,
    eagle3_warmup_sequence_lengths,
    warmup_eagle3_drafter,
    warmup_eagle3_tree_drafter,
)
from .eagle3_types import (
    Eagle3DraftCandidate,
    Eagle3DraftCandidatePool,
    Eagle3DraftTree,
    Eagle3GenerationResult,
    Eagle3TreeCycleMetrics,
    TreeCommitAudit,
    acceptance_prefix_length,
)
from .gpu_expert_cache import GpuExpertCache
from .lease import (
    ExpertCacheStats,
    ExpertLease,
    NoEvictableSlotError,
)
from .numeric_modes import (
    NUMERIC_MODES,
    NumericModeHandle,
    NumericSettings,
    apply_numeric_mode,
    current_numeric_settings,
    numeric_mode,
)
from .qwen3_offloaded_experts import (
    ExpertExecutionPolicy,
    Qwen3OffloadedExperts,
    ReductionMode,
)
from .qwen3_offloaded_model import (
    LoadedOffloadedQwen3,
    PackedExpertBackingMode,
    Qwen3LoadTimings,
    load_offloaded_qwen3,
)
from .qwen3_expert_prefetch import Qwen3ExpertPrefetch

__all__ = [
    "ExpertCacheStats",
    "ExpertExecutionPolicy",
    "ExpertLease",
    "AuxHiddenStateCollector",
    "CacheTransition",
    "CacheTransitionCounters",
    "CacheTransitionObserver",
    "CacheSlotSnapshot",
    "CumulativeProbabilityTreeSelector",
    "Eagle3ChainDrafter",
    "EAGLE3_DEFAULT_TREE_DEPTH",
    "EAGLE3_DEFAULT_TREE_NODE_BUDGET",
    "EAGLE3_DEFAULT_TREE_TOP_K",
    "EAGLE3_DEPTH5_MAX_DRAFTER_KV_TOKENS",
    "EAGLE3_DEPTH5_MAX_FED_TOKENS",
    "EAGLE3_MAX_COMMITTED_TOKENS",
    "EAGLE3_VALIDATED_TOTAL_TOKENS",
    "EAGLE3_VALIDATED_TREE_DEPTH",
    "Eagle3DraftCandidate",
    "Eagle3DraftCandidatePool",
    "Eagle3DraftTree",
    "Eagle3GenerationResult",
    "Eagle3TreeCycleMetrics",
    "Eagle3TreeDrafter",
    "ExplicitRawIdTreeSelector",
    "GpuExpertCache",
    "GpuExpertCacheSnapshot",
    "GreedyChainSpeculator",
    "GreedyChainTraceRecorder",
    "LoadedOffloadedQwen3",
    "LayerRouteMismatch",
    "LayerRoutingEquivalence",
    "NUMERIC_MODES",
    "NoEvictableSlotError",
    "NodeZeroNumericalReport",
    "NodeVerificationEquivalence",
    "NumericModeHandle",
    "NumericSettings",
    "PendingTransferSnapshot",
    "PackedExpertBackingMode",
    "PoolHindsightTreeSelector",
    "Qwen3OffloadedExperts",
    "Qwen3ExpertPrefetch",
    "Qwen3LoadTimings",
    "QWEN3_EAGLE3_LAYER_INDICES",
    "ReductionMode",
    "RawIdSelectionPolicy",
    "RecordingCacheTransitionObserver",
    "RuntimeEventKind",
    "ShallowTreeSpeculator",
    "TreeVerificationEquivalence",
    "TreeCommitAudit",
    "TreeSelectionDecision",
    "acceptance_prefix_length",
    "apply_numeric_mode",
    "aggregate_cache_transition_counters",
    "compare_tree_verification_to_causal_paths",
    "candidate_pool_fingerprint",
    "current_numeric_settings",
    "diagnose_tree_node_zero_numerics",
    "eagle3_tree_temporary_kv_tokens",
    "eagle3_tree_warmup_anchor_lengths",
    "eagle3_warmup_sequence_lengths",
    "load_offloaded_qwen3",
    "numeric_mode",
    "target_greedy_generate",
    "warmup_eagle3_drafter",
    "warmup_eagle3_tree_drafter",
]
