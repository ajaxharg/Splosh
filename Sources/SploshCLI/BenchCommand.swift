// BenchCommand.swift — SploshCLI.
//
// Owner: M0.5 creates the stub; M0.9 adds the flag grammar and the milestone named in rev4 §4.6
// makes the command real. Contract source: rev4 §4.6 (bench).

/// `splosh bench` — step timing at B=1..N over the contexts 2K/8K/32K/128K (rev4 §4.6).
public enum BenchCommand {
    public static let usage = """
        usage: splosh bench --context <C>[,<C>...] [options]

          --context <C>          2K | 8K | 32K | 128K, or a comma list
          --weights <path>       default $PWD/splosh-weights/
          --cache-dir <path>
          --restart-cycles <n>
          --speculative <on|off>
          --batch <n>            n in 1...16
          --concurrency <n>
          --generate <n>
          --agent-turn [--prefix <n>] [--thinking <n>]
          --greedy-agreement [--restore <entry>]
          --kv-format <int8|q4>  accepted only if M4.5 lands
          --help

        Every run prints toks_per_s and accepted_per_cycle together (rev4 C.1 rule R4).
        """

    /// Present but unimplemented: reports on stderr and exits non-zero (rev4 §6 M0.9).
    public static func run(_ arguments: [String]) -> Int32 {
        SploshCLI.notImplemented(.bench)
    }
}
