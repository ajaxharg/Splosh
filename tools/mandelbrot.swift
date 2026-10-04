#!/usr/bin/env swift
// mandelbrot.swift — render the Mandelbrot set to a PNG, in pure Swift.
//
// Escape-time iteration with smooth (continuous) coloring, plus a minimal hand-rolled
// PNG encoder (zlib via Foundation, CRC32 inline). No SPM, no CoreGraphics.
// Mirrors the CLI of tools/mandelbrot.py.
//
//   swift tools/mandelbrot.swift
//   swift tools/mandelbrot.swift --cx -0.744 --cy 0.114 --scale 0.05 --iter 4000 --palette ember
//   swiftc -O tools/mandelbrot.swift -o /tmp/mandelbrot && /tmp/mandelbrot --width 1920

import Foundation

// MARK: - Options

struct Options {
    var width = 1200
    var height = 900
    var cx = -0.6
    var cy = 0.0
    var scale = 3.0
    var maxIter = 512
    var bands = 8
    var palette = "classic"
    var output = "artifacts/mandelbrot-swift.png"
}

/// Cosine palettes: color(t) = a + b * cos(2*pi*(c*t + d)), t in [0, 1]; c is scaled by --bands.
let palettes: [String: (a: [Double], b: [Double], c: [Double], d: [Double])] = [
    "classic": ([0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1, 1, 1], [0.00, 0.10, 0.20]),
    "rainbow": ([0.50, 0.50, 0.50], [0.50, 0.50, 0.50], [1, 1, 1], [0.00, 0.33, 0.67]),
    "ember":   ([0.55, 0.40, 0.30], [0.45, 0.45, 0.35], [1, 1, 1], [0.00, 0.12, 0.28]),
    "ice":     ([0.35, 0.45, 0.55], [0.40, 0.40, 0.45], [1, 1, 1], [0.55, 0.60, 0.70]),
]

let usage = """
usage: swift tools/mandelbrot.swift [options]
  --width N    pixels (default 1200)
  --height N   pixels (default 900)
  --cx F       view center, real part (default -0.6)
  --cy F       view center, imaginary part (default 0)
  --scale F    real-axis span of the view (default 3.0)
  --iter N     max iterations (default 512)
  --bands N    color cycles across the iteration range (default 8)
  --palette P  classic|rainbow|ember|ice (default classic)
  --output F   PNG path (default artifacts/mandelbrot-swift.png)
"""

func parseOptions() -> Options? {
    var o = Options()
    let a = CommandLine.arguments
    var i = 1
    while i < a.count {
        guard i + 1 < a.count else { print(usage); return nil }
        let key = a[i], raw = a[i + 1]
        func setInt(what: String, target: inout Int) -> Bool {
            guard let n = Int(raw), n > 0 else {
                print("error: \(what) must be a positive integer, got \(raw)"); return false
            }
            target = n; return true
        }
        func setDouble(target: inout Double) -> Bool {
            guard let d = Double(raw) else {
                print("error: \(key) must be a number, got \(raw)"); return false
            }
            target = d; return true
        }
        switch key {
        case "--width":   guard setInt(what: "--width", target: &o.width) else { return nil }
        case "--height":  guard setInt(what: "--height", target: &o.height) else { return nil }
        case "--cx":      guard setDouble(target: &o.cx) else { return nil }
        case "--cy":      guard setDouble(target: &o.cy) else { return nil }
        case "--scale":
            guard setDouble(target: &o.scale), o.scale > 0 else {
                print("error: --scale must be a positive number"); return nil
            }
        case "--iter":    guard setInt(what: "--iter", target: &o.maxIter) else { return nil }
        case "--bands":   guard setInt(what: "--bands", target: &o.bands) else { return nil }
        case "--palette":
            guard palettes[raw] != nil else {
                print("error: unknown palette \(raw); choose from \(palettes.keys.sorted().joined(separator: ", "))")
                return nil
            }
            o.palette = raw
        case "--output":  o.output = raw
        default:
            print("error: unknown option \(key)"); print(usage); return nil
        }
        i += 2
    }
    return o
}

// MARK: - Escape-time render

func render(_ o: Options) -> (pixels: [UInt8], inSetPercent: Double, maxSmooth: Double) {
    let w = o.width, h = o.height
    var px = [UInt8](repeating: 0, count: w * h * 3)  // zero = interior black

    // Square pixels: the imaginary span scales with the aspect ratio.
    let x0 = o.cx - o.scale / 2
    let dx = o.scale / Double(w - 1)
    let yspan = o.scale * Double(h) / Double(w)
    let y0 = o.cy - yspan / 2
    let dy = yspan / Double(h - 1)
    let tScale = 1.0 / Double(max(1, o.maxIter))

    // 256-entry cosine palette LUT, indexed by the quantized smooth count.
    let pal = palettes[o.palette]!
    var lut = [UInt8](repeating: 0, count: 256 * 3)
    for q in 0..<256 {
        let t = Double(q) / 255.0
        for k in 0..<3 {
            let v = pal.a[k] + pal.b[k] * cos(2 * .pi * (pal.c[k] * Double(o.bands) * t + pal.d[k]))
            lut[q * 3 + k] = UInt8(255 * min(1, max(0, v)))
        }
    }

    var inSet = 0
    var maxSmooth = 0.0
    var p = 0
    for row in 0..<h {
        let ci = y0 + Double(row) * dy
        for col in 0..<w {
            let cr = x0 + Double(col) * dx
            var zr = 0.0, zi = 0.0
            var it = 0
            while it < o.maxIter {
                // Both coordinates must be computed from the previous iterate.
                let nr = zr * zr - zi * zi + cr
                zi = 2 * zr * zi + ci
                zr = nr
                it += 1
                if zr * zr + zi * zi > 4 { break }
            }
            if it < o.maxIter {
                // Just escaped: smooth count nu = n + 1 - log2(log|z_n|), n = iterations.
                let mag = zr * zr + zi * zi
                let smooth = Double(it) + 1 - log2(0.5 * log(mag))
                if smooth > maxSmooth { maxSmooth = smooth }
                let q = Int(min(1.0, smooth * tScale) * 255)
                px[p] = lut[q * 3]; px[p + 1] = lut[q * 3 + 1]; px[p + 2] = lut[q * 3 + 2]
            } else {
                inSet += 1
            }
            p += 3
        }
    }
    return (px, Double(inSet) / Double(w * h) * 100, maxSmooth)
}

// MARK: - Minimal PNG encoder (8-bit RGB, filter 0 per row)

@inline(__always) func appendBE32(_ v: UInt32, to data: inout Data) {
    var x = v.bigEndian
    withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
}

func crc32(_ bytes: Data) -> UInt32 {
    var table = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 {
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) == 1 ? (0xEDB88320 ^ (c >> 1)) : (c >> 1) }
        table[i] = c
    }
    var c: UInt32 = 0xFFFFFFFF
    for byte in bytes { c = table[Int((c ^ UInt32(byte)) & 0xFF)] ^ (c >> 8) }
    return c ^ 0xFFFFFFFF
}

func pngChunk(_ type: [UInt8], _ payload: Data) -> Data {
    var body = Data(type)
    body.append(payload)
    var chunk = Data()
    appendBE32(UInt32(payload.count), to: &chunk)
    chunk.append(contentsOf: body)
    appendBE32(crc32(body), to: &chunk)  // CRC covers type + payload
    return chunk
}

/// Adler-32 checksum of `bytes` (the trailer of a zlib stream).
func adler32(_ bytes: [UInt8]) -> UInt32 {
    var a: UInt32 = 1, b: UInt32 = 0
    for byte in bytes {
        a = (a + UInt32(byte)) % 65_521
        b = (b + a) % 65_521
    }
    return (b << 16) | a
}

func makePNG(width: Int, height: Int, rgb: [UInt8]) -> Data? {
    var ihdr = Data()
    appendBE32(UInt32(width), to: &ihdr)
    appendBE32(UInt32(height), to: &ihdr)
    ihdr.append(contentsOf: [8, 2, 0, 0, 0])  // 8 bits, color type 2 (RGB), no compression/filter/interlace

    let stride = width * 3 + 1
    var raw = [UInt8](repeating: 0, count: stride * height)  // filter byte 0 per row
    rgb.withUnsafeBytes { src in
        raw.withUnsafeMutableBytes { dst in
            // Raw pointers have no memcpy; bind to bytes and use the bound-pointer API.
            let s = src.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let d = dst.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for row in 0..<height {
                d.advanced(by: row * stride + 1)
                    .update(from: s.advanced(by: row * width * 3), count: width * 3)
            }
        }
    }
    // In this SDK `NSData.compressed(using: .zlib)` returns a *raw* deflate stream
    // (verified: it decodes cleanly as raw deflate — no zlib header, no Adler-32).
    // PNG requires the RFC 1950 envelope, so wrap it by hand:
    //   0x78 0x9C (deflate, 32K window) + raw deflate + big-endian Adler-32.
    guard let deflated = try? (Data(raw) as NSData).compressed(using: .zlib) else { return nil }
    var compressed = Data([0x78, 0x9C])
    compressed.append(deflated as Data)
    appendBE32(adler32(raw), to: &compressed)

    var out = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    out.append(pngChunk([0x49, 0x48, 0x44, 0x52], ihdr))
    out.append(pngChunk([0x49, 0x44, 0x41, 0x54], compressed))
    out.append(pngChunk([0x49, 0x45, 0x4E, 0x44], Data()))
    return out
}

// MARK: - Main

func main() {
    guard let o = parseOptions() else { exit(1) }
    let t0 = Date()
    let r = render(o)
    let t1 = Date()
    guard let png = makePNG(width: o.width, height: o.height, rgb: r.pixels) else {
        print("error: zlib compression failed"); exit(1)
    }
    let url = URL(fileURLWithPath: o.output)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                             withIntermediateDirectories: true)
    do {
        try png.write(to: url)
    } catch {
        print("error: could not write \(o.output): \(error)"); exit(1)
    }
    let t2 = Date()
    print("saved \(o.output) (\(o.width)x\(o.height))")
    print("  palette=\(o.palette) bands=\(o.bands) max_iter=\(o.maxIter)")
    print(String(format: "  render %.2fs  encode %.2fs  total %.2fs",
                 t1.timeIntervalSince(t0), t2.timeIntervalSince(t1), t2.timeIntervalSince(t0)))
    print(String(format: "  in-set pixels: %.1f%%   max smooth count: %.1f", r.inSetPercent, r.maxSmooth))
}

main()
