"""NanoFold Advanced Architecture Models.

This package contains the modular components of our novel protein folding
architecture, organized according to the 4-action paradigm:
    1. UNDERSTAND: Compact Micro-Pairformer trunk (pairformer.py)
    2. IMAGINE:    Global structural tokenizer and predictor (tokenizer.py)
    3. THINK:      Shared recurrent SE(3) structural workspace (reasoning.py)
    4. CONSTRUCT:  Stereochemical kinematics and Atom14 assembly (kinematics.py)
    5. REGULARIZE: Differentiable soft-core physics and clash penalties (physics.py)
"""

from .geometry_se3 import (
    RigidFrames,
    compute_fape_loss,
    make_extended_linear_chain,
    so3_exp_map,
)
from .kinematics import (
    Atom14KinematicsAssembly,
    KinematicsConfig,
    StereochemicalLoss,
    TorsionHead,
)
from .nanofold_model import (
    NanoFoldConfig,
    NanoFoldLoss,
    NanoFoldModel,
)
from .pairformer import (
    ContactHead,
    DistogramHead,
    InputEmbedder,
    MicroPairformer,
    PairformerBlock,
    PairformerConfig,
    TriangularMultiplicativeUpdate,
)
from .token_predictor import (
    PhaseCJEPALoss,
    PredictorConfig,
    TokenPredictor,
)
from .tokenizer import (
    GeometryAdapter,
    GlobalStructureTokenizer,
    StructureDecoder,
    StructureEncoder,
    TokenizerConfig,
)
from .workspace_thinker import (
    RecurrentWorkspaceThinker,
    WorkspaceReasoningBlock,
    WorkspaceThinkerConfig,
)

__all__ = [
    "ContactHead",
    "DistogramHead",
    "InputEmbedder",
    "MicroPairformer",
    "PairformerBlock",
    "PairformerConfig",
    "TriangularMultiplicativeUpdate",
    "GeometryAdapter",
    "GlobalStructureTokenizer",
    "StructureDecoder",
    "StructureEncoder",
    "TokenizerConfig",
    "PredictorConfig",
    "TokenPredictor",
    "PhaseCJEPALoss",
    "RigidFrames",
    "so3_exp_map",
    "make_extended_linear_chain",
    "compute_fape_loss",
    "WorkspaceThinkerConfig",
    "RecurrentWorkspaceThinker",
    "WorkspaceReasoningBlock",
    "KinematicsConfig",
    "TorsionHead",
    "Atom14KinematicsAssembly",
    "StereochemicalLoss",
    "NanoFoldConfig",
    "NanoFoldModel",
    "NanoFoldLoss",
]

