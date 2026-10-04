// SoakCommand.swift — SploshCLI.
//
// Owner: M0.5 creates the stub; M0.9 adds the flag grammar and the milestone named in rev4 §4.6
// makes the command real. Contract source: rev4 §4.6 (soak).

/// `splosh soak` — the sustained-load harness rev4 §7's soak gate runs.
public enum SoakCommand {
    public static let usage = """
        usage: splosh soak --minutes <n> --concurrency <n> --context <C> --generate <n> --report <path>

          --minutes <n>       total duration; one sample per minute
          --concurrency <n>   requests kept continuously in flight
          --context <C>       2K | 8K | 32K | 128K
          --generate <n>      tokens per request
          --report <path>     JSON, five fields per minute
          --help

        Exit 0 only when the final minute's decode rate is >= 0.90 x the first minute's and
        errors_cumulative == 0 (rev4 §4.6).
        """

    /// Present but unimplemented: reports on stderr and exits non-zero (rev4 §6 M0.9).
    public static func run(_ arguments: [String]) -> Int32 {
        SploshCLI.notImplemented(.soak)
    }
}
