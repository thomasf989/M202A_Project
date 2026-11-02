import Foundation

struct RoomInputFile {
    let url: URL
    let name: String
}

enum LocalRoomInput {
    static var workRoot: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RoomPreparation", isDirectory: true)
    }

    static func clearOldWork() throws {
        // Called only when the serial device worker starts, before any import.
        if FileManager.default.fileExists(atPath: workRoot.path) { try FileManager.default.removeItem(at: workRoot) }
        try FileManager.default.createDirectory(at: workRoot, withIntermediateDirectories: true)
    }

    static func stage(_ source: URL, model: Bool) throws -> RoomInputFile {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        let expected = model ? "glb" : "zip"
        guard source.pathExtension.lowercased() == expected else {
            throw RoomLocateError.message(model ? "请选择 .glb 三维房间模型。" : "请选择 .zip 参考数据。")
        }
        try FileManager.default.createDirectory(at: workRoot, withIntermediateDirectories: true)
        let target = workRoot.appendingPathComponent(UUID().uuidString + "." + expected)
        var complete = false
        defer { if !complete { try? FileManager.default.removeItem(at: target) } }
        let coordinator = NSFileCoordinator()
        var coordinationError: NSError?, readError: Error?
        coordinator.coordinate(readingItemAt: source, options: [], error: &coordinationError) { readable in
            do {
                let limit: UInt64 = model ? 192 * 1024 * 1024 : 4 * 1024 * 1024 * 1024
                let values = try readable.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard values.isRegularFile == true, let size = values.fileSize, size > 0, UInt64(size) <= limit else {
                    throw RoomLocateError.message(model ? "GLB 必须为完整文件，且不超过 192 MiB。" : "参考 ZIP 必须为完整文件，且不超过 4 GiB。")
                }
                try FileManager.default.copyItem(at: readable, to: target)
            } catch { readError = error }
        }
        if let coordinationError { throw coordinationError }
        if let readError { throw readError }
        guard FileManager.default.fileExists(atPath: target.path) else { throw RoomLocateError.message("文件未下载完成，请在“文件”App 中下载后重试。") }
        complete = true
        return RoomInputFile(url: target, name: source.lastPathComponent)
    }
}
