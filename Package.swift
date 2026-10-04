// swift-tools-version: 6.0
// Splosh — from-scratch Swift 6 / Metal 4 engine.
//
// Manifest authority: swift-engine-app-structure.rev4.md §3.1 (nine targets, dependency rules)
// and IMPLEMENTATION-PLAN.md §4.4 (SploshCLI takes a *direct* Runtime edge in addition to
// Server, Model, Quant, Core and Oracle, because Swift imports are not transitive and the
// composition root must instantiate Runtime's service and hand it to Server).
//
// Exactly one external dependency: Hummingbird, pinned to exactly 2.27.0 (rev4 §3.3). SwiftNIO
// arrives transitively. Package.resolved is committed so the resolution is reproducible.

import PackageDescription

let package = Package(
    name: "Splosh",
    platforms: [
        .macOS("27.0")
    ],
    // The executable's *product* is `splosh`, while its target stays `SploshCLI` (rev4 §3.1 names
    // exactly nine targets, and `SploshCLI` is one of them). Every CLI gate in the plan invokes
    // `swift run splosh ...`, and M0.5 asserts `test -x "$(swift build --show-bin-path)/splosh"`,
    // so the product name — not the target name — is what the build must produce.
    products: [
        .executable(name: "splosh", targets: ["SploshCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", exact: "2.27.0")
    ],
    targets: [
        // Core depends on nothing (rev4 §3.1 rule 1).
        //
        // `default.metallib` is a build product of `make shaders` (rev4 §3.3) and is git-ignored.
        // `.copy` puts it in the bundle root, where `Metallib` finds it through `Bundle.module`.
        // The Makefile must therefore run BEFORE `swift build`: SwiftPM validates the declared
        // resource during manifest loading, so a missing metallib fails the build rather than
        // producing a library that loads to nothing.
        .target(
            name: "SploshCore",
            resources: [.copy("Resources/default.metallib")]
        ),

        .target(
            name: "SploshQuant",
            dependencies: ["SploshCore"]
        ),

        .target(
            name: "SploshModel",
            dependencies: ["SploshCore", "SploshQuant"]
        ),

        // Oracle stays a scalar reference: Model, Quant, Core only — never Runtime or Server
        // (rev4 §3.1 rule 5).
        .target(
            name: "SploshOracle",
            dependencies: ["SploshModel", "SploshQuant", "SploshCore"]
        ),

        // Runtime must never import Server (rev4 §3.1 rule 2).
        .target(
            name: "SploshRuntime",
            dependencies: ["SploshModel", "SploshCore"]
        ),

        .target(
            name: "SploshServer",
            dependencies: [
                "SploshRuntime",
                "SploshModel",
                "SploshCore",
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),

        // The composition root. Runtime is listed explicitly per IMPLEMENTATION-PLAN.md §4.4.
        .executableTarget(
            name: "SploshCLI",
            dependencies: [
                "SploshRuntime",
                "SploshServer",
                "SploshModel",
                "SploshQuant",
                "SploshCore",
                "SploshOracle",
            ]
        ),

        .testTarget(
            name: "SploshOracleTests",
            dependencies: ["SploshModel", "SploshQuant", "SploshCore", "SploshRuntime", "SploshOracle"]
        ),

        // The one target allowed to import SploshCLI (rev4 §3.1 rule 3: M0.9's argv table).
        .testTarget(
            name: "SploshServerTests",
            dependencies: ["SploshServer", "SploshRuntime", "SploshCLI"]
        ),
    ]
)
