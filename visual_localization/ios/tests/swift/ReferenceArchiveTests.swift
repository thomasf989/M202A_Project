import Foundation
import XCTest
import ZIPFoundation
@testable import RoomArchive

final class ReferenceArchiveTests: XCTestCase {
    private var root: URL!
    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try FileManager.default.removeItem(at: root) }

    private func archive(_ entries: [(String, Data, Entry.EntryType)], named: String = "input.zip") throws -> URL {
        let path = root.appendingPathComponent(named)
        let archive = try Archive(url: path, accessMode: .create)
        for (name, data, type) in entries {
            try archive.addEntry(with: name, type: type, uncompressedSize: Int64(data.count), compressionMethod: .none) { position, size in
                data.subdata(in: Int(position)..<min(data.count, Int(position) + size))
            }
        }
        return path
    }
    private func extract(_ source: URL, control: Progress = Progress(totalUnitCount: 100)) throws -> URL {
        let output = root.appendingPathComponent("output", isDirectory: true)
        try ReferenceArchive.extract(source, to: output, cancellation: control) { _, _ in }
        return output
    }

    func testValidArchiveKeepsMetadataAndPhotos() throws {
        let source = try archive([("scene/references.json", Data("{}".utf8), .file),
                                  ("scene/ref.png", Data([1, 2, 3]), .file),
                                  ("__MACOSX/._ref.png", Data([0]), .file)])
        let output = try extract(source)
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("scene/ref.png")), Data([1, 2, 3]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("__MACOSX").path))
    }
    func testTraversalAndSymbolicLinksAreRejectedAndCleaned() throws {
        for (index, entry) in [("../escape.json", Data("{}".utf8), Entry.EntryType.file),
                               ("link.json", Data("../outside".utf8), Entry.EntryType.symlink)].enumerated() {
            let source = try archive([entry], named: "bad-\(index).zip")
            XCTAssertThrowsError(try extract(source))
            XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("output").path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: root.deletingLastPathComponent().appendingPathComponent("escape.json").path))
        }
    }
    func testCaseConflictsAndCRCFailuresDoNotLeavePartialData() throws {
        let duplicate = try archive([("Frame.json", Data([1]), .file), ("frame.json", Data([2]), .file)])
        XCTAssertThrowsError(try extract(duplicate))
        let corrupt = try archive([("references.json", Data("room-example".utf8), .file)], named: "corrupt.zip")
        var bytes = try Data(contentsOf: corrupt)
        let range = try XCTUnwrap(bytes.range(of: Data("room-example".utf8)))
        bytes[range.lowerBound] ^= 1
        try bytes.write(to: corrupt)
        XCTAssertThrowsError(try extract(corrupt))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("output").path))
    }
    func testCancellationAndExistingDestinationPreserveSavedData() throws {
        let source = try archive([("reference.json", Data([1]), .file)])
        let control = Progress(totalUnitCount: 100); control.cancel()
        XCTAssertThrowsError(try extract(source, control: control))
        let output = root.appendingPathComponent("output", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let saved = output.appendingPathComponent("keep.txt"); try Data("saved".utf8).write(to: saved)
        XCTAssertThrowsError(try extract(source))
        XCTAssertEqual(try String(contentsOf: saved, encoding: .utf8), "saved")
    }
    func testOfficialNestedArchiveExpandsOnlyTrainingSequences() throws {
        let train = try archive([("seq-01/frame-000000.color.png", Data([1, 2]), .file),
                                 ("seq-01/frame-000000.depth.png", Data([8]), .file)], named: "train.zip")
        let test = try archive([("seq-03/frame-000000.color.png", Data([9]), .file)], named: "test.zip")
        let source = try archive([("Fire/TrainSplit.txt", Data("sequence1\n".utf8), .file),
                                  ("Fire/TestSplit.txt", Data("sequence3\n".utf8), .file),
                                  ("Fire/seq-01.zip", try Data(contentsOf: train), .file),
                                  ("Fire/seq-03.zip", try Data(contentsOf: test), .file)], named: "scene.zip")
        let output = try extract(source)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("Fire/seq-01/frame-000000.color.png").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("Fire/seq-03.zip").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("Fire/seq-03").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("Fire/seq-01/frame-000000.depth.png").path))
    }
    func testOversizedSplitRejectedBeforeReadingItsContents() throws {
        let source = try archive([("TrainSplit.txt", Data(repeating: 65, count: 65_537), .file)])
        XCTAssertThrowsError(try extract(source))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("output").path))
    }
}
