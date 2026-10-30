import Foundation
import SwiftUI

private struct LoadedRoom {
    let record: RoomRecord
    let mesh: RoomMesh
    let calibration: Calibration?
}

// OpenCV instances and all map I/O stay on this single background queue.
// No network API, server address, Python runtime, or model download is used.
private final class DeviceWorker: @unchecked Sendable {
    private let queue = DispatchQueue(label: "roomloc.device", qos: .userInitiated)
    private var native: RLNativeLocalizer?
    private var current: RoomRecord?

    private func perform<T>(_ operation: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try operation()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }
    func rooms() async throws -> [RoomRecord] { try await perform { try RoomStore.list() } }
    func importRoom(_ url: URL) async throws -> LoadedRoom {
        try await perform {
            let (record, engine) = try RoomStore.importPackage(url)
            let mesh = try RoomMesh.load(at: RoomStore.directory(record.id).appendingPathComponent("mesh.bin"))
            self.native = engine; self.current = record
            return LoadedRoom(record: record, mesh: mesh, calibration: RoomStore.calibration(record))
        }
    }
    func load(_ record: RoomRecord) async throws -> LoadedRoom {
        try await perform {
            let folder = RoomStore.directory(record.id)
            let mesh = try RoomMesh.load(at: folder.appendingPathComponent("mesh.bin"))
            let engine: RLNativeLocalizer? = try RLNativeLocalizer(mapPath: folder.appendingPathComponent("map.yml").path)
            guard let engine else { throw RoomLocateError.message("参考库加载失败。") }
            self.native = engine; self.current = record
            return LoadedRoom(record: record, mesh: mesh, calibration: RoomStore.calibration(record))
        }
    }
    func preparePhoto(_ data: Data) async throws -> PreparedPhoto { try await perform { try PreparedPhoto.prepare(data) } }
    func saveCalibration(_ url: URL, room: RoomRecord) async throws -> Calibration {
        try await perform { try RoomStore.saveCalibration(url, room: room) }
    }
    func localize(_ photo: PreparedPhoto, mode: CameraMode, calibration: Calibration?) async throws -> (LocalizationResult, URL) {
        try await perform {
            guard let native = self.native, let room = self.current else { throw RoomLocateError.message("请先导入或选择房间。") }
            let text: String? = try native.localizeJPEG(photo.jpeg, cameraJSON: photo.cameraJSON(mode: mode, calibration: calibration))
            guard let text, let data = text.data(using: .utf8) else { throw RoomLocateError.message("定位结果读取失败。") }
            let result = try JSONDecoder().decode(LocalizationResult.self, from: data)
            var saved = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            saved["room_id"] = room.id.uuidString; saved["room_name"] = room.name
            saved["timestamp"] = ISO8601DateFormatter().string(from: Date())
            saved["original_image_size"] = [photo.originalWidth, photo.originalHeight]
            saved["processed_image_size"] = [Int(photo.preview.size.width), Int(photo.preview.size.height)]
            let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Results", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let url = root.appendingPathComponent("pose-\(UUID().uuidString).json")
            try JSONSerialization.data(withJSONObject: saved, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
            return (result, url)
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var rooms = [RoomRecord]()
    @Published private(set) var selectedRoom: RoomRecord?
    @Published private(set) var mesh: RoomMesh?
    @Published private(set) var photo: PreparedPhoto?
    @Published private(set) var calibration: Calibration?
    @Published private(set) var result: LocalizationResult?
    @Published private(set) var resultURL: URL?
    @Published private(set) var busy = false
    @Published var message = "导入房间包后，即可在手机上离线定位。"
    @Published var error: String?
    @Published var cameraMode: CameraMode = .auto
    private let worker = DeviceWorker()
    private var started = false

    func start() async {
        guard !started else { return }; started = true
        do {
            rooms = try await worker.rooms()
            let remembered = UserDefaults.standard.string(forKey: "selectedRoom")
            if let room = rooms.first(where: { $0.id.uuidString == remembered }) ?? rooms.first { await select(room) }
        } catch { self.error = error.localizedDescription }
    }
    private func clearResult() { result = nil; resultURL = nil }
    private func accept(_ loaded: LoadedRoom) {
        selectedRoom = loaded.record; mesh = loaded.mesh; calibration = loaded.calibration
        cameraMode = .auto; photo = nil; clearResult()
        UserDefaults.standard.set(loaded.record.id.uuidString, forKey: "selectedRoom")
        message = "房间已就绪 · \(loaded.record.references) 张参考图 · 本机计算"
    }
    func select(_ room: RoomRecord) async {
        guard !busy else { return }; busy = true; error = nil; clearResult(); message = "正在加载房间…"
        defer { busy = false }
        do { accept(try await worker.load(room)) }
        catch { self.error = error.localizedDescription; message = "房间加载失败，请重新选择或导入。" }
    }
    func importRoom(_ url: URL) async {
        guard !busy else { error = "请等待当前任务完成后，再打开房间包。"; return }
        busy = true; error = nil; message = "正在导入和检查房间包…"
        defer { busy = false }
        do {
            let loaded = try await worker.importRoom(url)
            rooms = try await worker.rooms(); accept(loaded)
        } catch { self.error = error.localizedDescription; message = "导入失败，请检查房间包。" }
    }
    func choosePhoto(_ data: Data) async {
        guard !busy else { return }; busy = true; error = nil; clearResult(); photo = nil
        message = "正在准备照片…"; defer { busy = false }
        do { photo = try await worker.preparePhoto(data); message = "照片已准备好，可以开始定位。" }
        catch { self.error = error.localizedDescription; message = "照片读取失败。" }
    }
    func importCalibration(_ url: URL) async {
        guard !busy, let room = selectedRoom else { return }; busy = true; error = nil; clearResult()
        defer { busy = false }
        do { calibration = try await worker.saveCalibration(url, room: room); cameraMode = .saved; message = "标定已保存，仅对相同镜头、裁剪和方向复用。" }
        catch { self.error = error.localizedDescription }
    }
    func localize() async {
        guard !busy, let photo, selectedRoom != nil else { return }
        busy = true; error = nil; clearResult(); message = "正在手机上匹配特征并计算位姿…"
        defer { busy = false }
        do {
            let (value, url) = try await worker.localize(photo, mode: cameraMode, calibration: calibration)
            result = value; resultURL = url
            message = value.success ? "已返回拍照位姿，红色相机显示位置与朝向。" : value.failureMessage
        } catch { self.error = error.localizedDescription; message = "定位未完成。" }
    }
    func cameraModeChanged() { clearResult() }
}
