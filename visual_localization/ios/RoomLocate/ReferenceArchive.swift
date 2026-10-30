import Foundation
import ZIPFoundation

struct ReferenceImportError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// This layer has no UIKit or network dependency. Entries are streamed into a
// private staging directory; archive paths never select an external location.
enum ReferenceArchive {
    static let maximumExpandedBytes: UInt64 = 6 * 1024 * 1024 * 1024
    static let maximumEntries = 80_000

    private final class Budget {
        var bytes: UInt64 = 0
        var entries = 0
        var paths = Set<String>()
    }

    static func extract(_ source: URL, to destination: URL, cancellation: Progress,
                        report: (Double, String) -> Void) throws {
        let manager = FileManager.default
        guard !manager.fileExists(atPath: destination.path) else {
            throw ReferenceImportError(message: "请使用新的参考数据准备目录。")
        }
        try manager.createDirectory(at: destination, withIntermediateDirectories: true)
        var complete = false
        defer { if !complete { try? manager.removeItem(at: destination) } }
        let budget = Budget()
        try unpack(source, to: destination, budget: budget, cancellation: cancellation, report: report)

        // Official outer scene archives contain seq-XX.zip. Only official
        // training sequence archives are expanded, never test sequence ZIPs.
        let enumerator = manager.enumerator(at: destination, includingPropertiesForKeys: [.isRegularFileKey])
        let splits = (enumerator?.allObjects as? [URL] ?? []).filter { $0.lastPathComponent == "TrainSplit.txt" }
        if splits.count == 1, let train = splits.first {
            let sequences = try sequenceNames(String(contentsOf: train, encoding: .utf8))
            for (index, sequence) in sequences.enumerated() {
                try check(cancellation)
                let nested = train.deletingLastPathComponent().appendingPathComponent(sequence + ".zip")
                if manager.fileExists(atPath: nested.path) {
                    report(0.5, "正在解包训练序列 \(index + 1)/\(sequences.count)…")
                    try unpack(nested, to: train.deletingLastPathComponent(), budget: budget,
                               cancellation: cancellation, report: { _, text in report(0.5, text) })
                    try manager.removeItem(at: nested)
                }
            }
        }
        try check(cancellation)
        complete = true
        report(1, "参考文件已解包，正在检查照片和位姿…")
    }

    static func check(_ cancellation: Progress) throws {
        if cancellation.isCancelled { throw CancellationError() }
    }

    private static func sequenceNames(_ text: String) throws -> [String] {
        let rows = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !rows.isEmpty, rows.count <= 1000 else {
            throw ReferenceImportError(message: "7-Scenes 划分文件为空或过大。")
        }
        return try rows.map { row in
            guard row.range(of: "^sequence[0-9]{1,4}$", options: .regularExpression) != nil,
                  let value = Int(row.dropFirst(8)) else {
                throw ReferenceImportError(message: "7-Scenes 官方划分文件无效。")
            }
            return String(format: "seq-%02d", value)
        }
    }

    private static func components(_ path: String) throws -> [String] {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !path.hasPrefix("/"), !path.contains("\\"), !path.contains("\0"),
              !path.contains(":"), !parts.contains(".."), !parts.contains("."),
              !parts.dropLast().contains("") else {
            throw ReferenceImportError(message: "参考 ZIP 包含不安全的文件路径。")
        }
        return parts.filter { !$0.isEmpty }
    }

    private static func unpack(_ source: URL, to destination: URL, budget: Budget,
                               cancellation: Progress, report: (Double, String) -> Void) throws {
        let archive = try Archive(url: source, accessMode: .read)
        // Reading the small official split first lets us skip test directories
        // and test ZIPs even when they are present in the selected archive.
        var entries = [Entry]()
        for entry in archive {
            try check(cancellation)
            budget.entries += 1
            guard budget.entries <= maximumEntries else {
                throw ReferenceImportError(message: "参考 ZIP 文件数量过多，请选择单个房间的数据。")
            }
            let parts = try components(entry.path)
            guard entry.type != .symlink else {
                throw ReferenceImportError(message: "参考 ZIP 不能包含符号链接。")
            }
            if parts.contains(where: { $0.hasPrefix(".") || $0 == "__MACOSX" }) { continue }
            entries.append(entry)
        }
        let trainFiles = entries.filter { ($0.path as NSString).lastPathComponent == "TrainSplit.txt" && $0.type == .file }
        var splitPrefix: String?
        var allowedSequences = Set<String>()
        if trainFiles.count == 1, let train = trainFiles.first {
            guard train.uncompressedSize <= 65_536 else {
                throw ReferenceImportError(message: "7-Scenes 划分文件大小无效。")
            }
            var data = Data()
            let crc = try archive.extract(train, bufferSize: 65_536) { chunk in
                try check(cancellation)
                guard data.count + chunk.count <= 65_536 else { throw ReferenceImportError(message: "7-Scenes 划分文件过大。") }
                data.append(chunk)
            }
            guard crc == train.checksum, let text = String(data: data, encoding: .utf8) else {
                throw ReferenceImportError(message: "7-Scenes 划分文件损坏。")
            }
            allowedSequences = Set(try sequenceNames(text))
            splitPrefix = String(train.path.dropLast("TrainSplit.txt".count))
        }
        for (index, entry) in entries.enumerated() {
            try check(cancellation)
            guard entry.type == .file else { continue }
            if let prefix = splitPrefix, entry.path.hasPrefix(prefix) {
                let tail = String(entry.path.dropFirst(prefix.count))
                let first = tail.split(separator: "/").first.map(String.init) ?? ""
                let sequence = first.hasSuffix(".zip") ? String(first.dropLast(4)) : first
                if sequence.range(of: "^seq-[0-9]+$", options: .regularExpression) != nil,
                   !allowedSequences.contains(sequence) { continue }
            }
            // Reference depth is rendered from GLB in the RGB camera grid.
            // Unregistered Kinect depth and Polycam sensor maps aren't needed.
            if entry.path.hasSuffix(".depth.png") || entry.path.contains("keyframes/depth/") || entry.path.contains("keyframes/confidence/") { continue }
            let ext = (entry.path as NSString).pathExtension.lowercased()
            guard ["json", "txt", "png", "jpg", "jpeg", "zip"].contains(ext) else { continue }
            let limit: UInt64 = ext == "zip" ? 2 * 1024 * 1024 * 1024 : 256 * 1024 * 1024
            guard entry.uncompressedSize <= limit,
                  entry.uncompressedSize <= maximumExpandedBytes - budget.bytes else {
                throw ReferenceImportError(message: "参考数据解包超过手机限制，请减少输入照片或序列。")
            }
            let target = destination.appendingPathComponent(entry.path).standardizedFileURL
            guard target.path.hasPrefix(destination.standardizedFileURL.path + "/"),
                  budget.paths.insert(target.path.precomposedStringWithCanonicalMapping.lowercased()).inserted,
                  !FileManager.default.fileExists(atPath: target.path) else {
                throw ReferenceImportError(message: "参考 ZIP 包含重复或冲突的文件路径。")
            }
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            guard FileManager.default.createFile(atPath: target.path, contents: nil) else {
                throw ReferenceImportError(message: "参考文件保存失败，请检查手机剩余空间。")
            }
            let file = try FileHandle(forWritingTo: target)
            defer { try? file.close() }
            var actual: UInt64 = 0
            let crc = try archive.extract(entry, bufferSize: 256 * 1024) { chunk in
                try check(cancellation)
                guard UInt64(chunk.count) <= entry.uncompressedSize - actual,
                      UInt64(chunk.count) <= maximumExpandedBytes - budget.bytes else {
                    throw ReferenceImportError(message: "参考 ZIP 解包长度无效。")
                }
                try file.write(contentsOf: chunk)
                actual += UInt64(chunk.count); budget.bytes += UInt64(chunk.count)
            }
            guard actual == entry.uncompressedSize, crc == entry.checksum else {
                throw ReferenceImportError(message: "参考 ZIP 校验失败，请重新下载完整文件。")
            }
            if index % 32 == 0 { report(Double(index + 1) / Double(max(1, entries.count)) * 0.5, "正在解包参考数据…") }
        }
    }
}
