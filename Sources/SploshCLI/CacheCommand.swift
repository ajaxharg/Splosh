// CacheCommand.swift — SploshCLI.
//
// `splosh cache`: look at, and empty, the on-disk prefix store that `splosh serve` writes
// (SploshRuntime/PrefixStore.swift).

import Foundation
import SploshRuntime

public enum CacheCommand {
    public static let usage = """
        usage: splosh cache <stats|path|purge> [--dir <path>] [--all]

          stats   read-only: the stored prefixes, newest first, with tokens and bytes
          path    prints the resolved store directory
          purge   deletes stored prefixes; requires --all
          --dir   the store directory; default is prefixCacheDir from ./splosh.toml,
                  else ~/Library/Caches/Splosh/prefix-cache
          --help
        """

    public static func run(_ arguments: [String]) -> Int32 {
        var action: String?
        var directory: String?
        var all = false
        var index = 0
        while index < arguments.count {
            let token = arguments[index]
            switch token {
            case "--help", "-h": print(usage); return ExitStatus.ok
            case "--all": all = true
            case "--dir":
                guard index + 1 < arguments.count else { return fail("--dir needs a path") }
                index += 1
                directory = arguments[index]
            case "stats", "path", "purge":
                guard action == nil else { return fail("more than one action given") }
                action = token
            default: return fail("unexpected argument '\(token)'")
            }
            index += 1
        }
        guard let action else { SploshCLI.writeStderr(usage + "\n"); return ExitStatus.usage }
        let configured = directory ?? ((try? ServeConfig.load())?.prefixCacheDir ?? ServeConfig().prefixCacheDir)
        guard configured != "none" else { return fail("the prefix store is disabled (prefixCacheDir = \"none\")") }
        let url = URL(fileURLWithPath: (configured as NSString).expandingTildeInPath, isDirectory: true)
        let items = PrefixStore.inventory(directory: url)
        func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / 1_073_741_824) }
        switch action {
        case "path":
            print(url.path)
        case "stats":
            print("\(url.path): \(items.count) stored, \(gib(items.reduce(0) { $0 + $1.bytes }))")
            let formatter = ISO8601DateFormatter()
            for item in items {
                print(String(format: "  %8d tokens  %9@  %@  %@  %@", item.tokenCount, gib(item.bytes) as NSString,
                             formatter.string(from: item.modified) as NSString, (item.stateOnly ? "checkpoint" : "prefix    ") as NSString,
                             item.url.lastPathComponent as NSString))
            }
        default:
            guard all else { return fail("purge deletes every stored prefix; pass --all to confirm") }
            var removed = 0, bytes = 0
            for item in items where (try? FileManager.default.removeItem(at: item.url)) != nil { removed += 1; bytes += item.bytes }
            print("removed \(removed) stored prefixes, \(gib(bytes))")
        }
        return ExitStatus.ok
    }

    private static func fail(_ message: String) -> Int32 {
        SploshCLI.writeStderr("cache: \(message)\n")
        return ExitStatus.usage
    }
}
