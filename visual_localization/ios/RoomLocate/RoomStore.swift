import Foundation

// A room package is a bounded flat container, not an executable or a ZIP.
// No package-controlled path is ever written to the filesystem.
enum RoomStore {
    static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Rooms", isDirectory: true)
    }
    static func directory(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }

    static func build(glb: URL, references: URL, name: String, maxReferences: Int,
                      cancellation: Progress, report: @escaping (Double, String) -> Void) throws -> (RoomRecord, RLNativeLocalizer) {
        let work = LocalRoomInput.workRoot.appendingPathComponent("build-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: work) }
        let referenceRoot = work.appendingPathComponent("references", isDirectory: true)
        try ReferenceArchive.extract(references, to: referenceRoot, cancellation: cancellation) { value, text in report(value * 0.2, text) }
        let prepared = work.appendingPathComponent("prepared", isDirectory: true)
        let metadataJSON: String? = try RLNativeLocalizer.buildRoom(
            glbPath: glb.path, referenceRoot: referenceRoot.path, outputDirectory: prepared.path,
            name: name, maxReferences: maxReferences, progress: { value, text in
                if value >= 0 { report(0.2 + value * 0.78, text) }
                return !cancellation.isCancelled
            })
        try ReferenceArchive.check(cancellation)
        guard let metadataJSON else { throw RoomLocateError.message("房间参考库生成失败。") }
        try RoomMesh.validate(at: prepared.appendingPathComponent("mesh.bin"))
        let engine: RLNativeLocalizer? = try RLNativeLocalizer(mapPath: prepared.appendingPathComponent("map.yml").path)
        guard let engine else { throw RoomLocateError.message("新房间参考库验证失败。") }
        let metadata = try JSONDecoder().decode(NativeMetadata.self, from: Data(metadataJSON.utf8))
        let record = RoomRecord(id: UUID(), name: metadata.name, references: metadata.references,
                                datasetFormat: metadata.dataset_format, datasetCameraJSON: metadata.dataset_camera_json)
        try JSONEncoder().encode(record).write(to: prepared.appendingPathComponent("record.json"), options: .atomic)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try ReferenceArchive.check(cancellation)
        try FileManager.default.moveItem(at: prepared, to: directory(record.id))
        report(1, "房间已保存，可直接拍照定位。")
        return (record, engine)
    }

    static func list() throws -> [RoomRecord] {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .compactMap { folder in
                guard let id = UUID(uuidString: folder.lastPathComponent),
                      let data = try? Data(contentsOf: folder.appendingPathComponent("record.json")),
                      let record = try? JSONDecoder().decode(RoomRecord.self, from: data), record.id == id else { return nil }
                return record
            }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    static func importPackage(_ source: URL) throws -> (RoomRecord, RLNativeLocalizer) {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let header = try input.read(upToCount: 24) ?? Data()
        guard header.count == 24, String(data: header.prefix(8), encoding: .ascii) == "RLOC0001" else {
            throw RoomLocateError.message("请选择 .roomloc 房间包。若要导入 GLB 和参考 ZIP，请使用“新建房间”。")
        }
        func little64(_ offset: Int) -> UInt64 {
            header[offset..<(offset + 8)].enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * $1.offset)) }
        }
        let mapSize = little64(8), meshSize = little64(16)
        let total = try input.seekToEnd()
        guard mapSize > 0, mapSize <= 256 * 1024 * 1024,
              meshSize >= 16, meshSize <= 64 * 1024 * 1024,
              total == 24 + mapSize + meshSize else {
            throw RoomLocateError.message("房间包大小无效、超出手机限制或尚未传输完整。")
        }
        try input.seek(toOffset: 24)
        let id = UUID(), folder = directory(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var committed = false
        defer { if !committed { try? FileManager.default.removeItem(at: folder) } }
        for (name, size) in [("map.yml", mapSize), ("mesh.bin", meshSize)] {
            let path = folder.appendingPathComponent(name)
            guard FileManager.default.createFile(atPath: path.path, contents: nil) else {
                throw RoomLocateError.message("无法保存房间包，请检查手机可用空间。")
            }
            let output = try FileHandle(forWritingTo: path)
            defer { try? output.close() }
            var remaining = size
            while remaining > 0 {
                let chunk = try input.read(upToCount: Int(min(remaining, 1024 * 1024))) ?? Data()
                guard !chunk.isEmpty else { throw RoomLocateError.message("房间包传输中断。") }
                try output.write(contentsOf: chunk); remaining -= UInt64(chunk.count)
            }
            try output.synchronize()
        }
        // Both the feature map and display geometry are validated before the
        // registry becomes visible to the UI or a previous room is replaced.
        try RoomMesh.validate(at: folder.appendingPathComponent("mesh.bin"))
        let engine: RLNativeLocalizer? = try RLNativeLocalizer(mapPath: folder.appendingPathComponent("map.yml").path)
        guard let engine else { throw RoomLocateError.message("房间参考库无法读取。") }
        let metadata = try JSONDecoder().decode(NativeMetadata.self, from: Data(engine.metadataJSON().utf8))
        let record = RoomRecord(id: id, name: metadata.name, references: metadata.references,
                                datasetFormat: metadata.dataset_format, datasetCameraJSON: metadata.dataset_camera_json)
        try JSONEncoder().encode(record).write(to: folder.appendingPathComponent("record.json"), options: .atomic)
        committed = true
        return (record, engine)
    }

    static func calibration(_ room: RoomRecord) -> Calibration? {
        guard let data = try? Data(contentsOf: directory(room.id).appendingPathComponent("camera.json")),
              let value = try? JSONDecoder().decode(Calibration.self, from: data),
              (try? value.validate()) != nil else { return nil }
        return value
    }

    static func saveCalibration(_ url: URL, room: RoomRecord) throws -> Calibration {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0, size <= 65536 else { throw RoomLocateError.message("标定 JSON 文件大小无效。") }
        let value = try JSONDecoder().decode(Calibration.self, from: Data(contentsOf: url))
        try value.validate()
        try JSONEncoder().encode(value).write(to: directory(room.id).appendingPathComponent("camera.json"), options: .atomic)
        return value
    }
}
