import Metal

/// The dispatches of the candidate gate/up/SiLU stage (SPLOSH_MLP_FUSED, see
/// Sources/Shaders/candidates/mlp_fused.metal) for one step: which kernels, over which rows,
/// on which grid. The engine and the kernels' test both encode from this, so the geometry the
/// test checks is the geometry the engine dispatches.
///
/// Variants (the value of SPLOSH_MLP_FUSED):
///   1  fp32 gate, per-64 sums read back from the stored tile
///   2  fp32 gate, sums from registers
///   3  bf16 gate, sums read back
///   4  bf16 gate, sums from registers
/// A step whose last 32-row tile is partial has no bf16 gate (variants 3 and 4 then run as 1 and
/// 2), and its partial tile is a dispatch of its own whose sums are always read back.
public struct MlpFusedPlan {
    public static let variants = 1...4

    public struct Dispatch {
        public let kernel: String
        /// The kernel's row bound, and the first 32-row tile the grid covers.
        public let rows: Int, firstTile: Int
        public let grid: MTLSize
    }

    /// The gate GEMM storing bf16, over every tile of the step; nil when the gate is the shipped
    /// wide GEMM's fp32 output.
    public let gate: Dispatch?
    /// The up GEMM with the product in its epilogue. Each reads the gate, so a barrier
    /// separates these from the gate GEMM.
    public let up: [Dispatch]

    public static let threadsPerGroup = MTLSize(width: 128, height: 1, depth: 1)

    /// Every kernel a variant can dispatch.
    public static func kernels(variant: Int) -> [String] {
        let registers = variant == 2 || variant == 4, bf16 = variant >= 3
        var names = ["sp_mlp_up_silu_m32n128s4_partial", "sp_mlp_up_silu_m32n128s4" + (registers ? "_regsums" : "")]
        if bf16 { names += ["sp_mlp_gate_bf16_m32n128s4", "sp_mlp_up_silu_m32n128s4_bgate" + (registers ? "_regsums" : "")] }
        return names
    }

    /// Row tiles go four to a block along the grid's x, blocks along z (see SP_MLP_ROW_BLOCK).
    public static func grid(rowTiles: Int, outDim: Int) -> MTLSize {
        MTLSize(width: min(rowTiles, 4), height: outDim / 128, depth: (rowTiles + 3) / 4)
    }

    public init?(variant: Int, rows: Int, outDim: Int) {
        guard Self.variants.contains(variant), rows > 0, outDim % 128 == 0 else { return nil }
        let wholeTiles = rows / 32, partial = rows % 32 != 0
        let registers = variant == 2 || variant == 4, bf16 = variant >= 3 && !partial
        let wholeGrid = Self.grid(rowTiles: wholeTiles, outDim: outDim)
        gate = bf16 ? Dispatch(kernel: "sp_mlp_gate_bf16_m32n128s4", rows: rows, firstTile: 0, grid: wholeGrid) : nil
        var dispatches: [Dispatch] = []
        if wholeTiles > 0 {
            dispatches.append(Dispatch(kernel: "sp_mlp_up_silu_m32n128s4" + (bf16 ? "_bgate" : "") + (registers ? "_regsums" : ""),
                                       rows: wholeTiles * 32, firstTile: 0, grid: wholeGrid))
        }
        if partial {
            dispatches.append(Dispatch(kernel: "sp_mlp_up_silu_m32n128s4_partial", rows: rows, firstTile: wholeTiles,
                                       grid: Self.grid(rowTiles: 1, outDim: outDim)))
        }
        up = dispatches
    }
}
