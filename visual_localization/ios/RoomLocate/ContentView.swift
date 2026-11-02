import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct ContentView: View {
    @ObservedObject var model: AppModel
    @State private var importRoom = false
    @State private var createRoom = false
    @State private var importCalibration = false
    @State private var importPhoto = false
    @State private var showCamera = false
    @State private var selection: PhotosPickerItem?
    @State private var cameraReset = 0

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Label("手机独立运行 · 照片仅在本机处理", systemImage: "iphone")
                        .font(.subheadline).foregroundStyle(.secondary)
                    roomControls
                    ZStack {
                        RoomSceneView(roomID: model.selectedRoom?.id, mesh: model.mesh, pose: model.result?.pose, reset: cameraReset)
                        if model.mesh == nil {
                            VStack(spacing: 10) {
                                Image(systemName: "cube.transparent").font(.largeTitle)
                                Text("导入 GLB 和参考数据，建立房间")
                            }.foregroundStyle(.secondary)
                        }
                    }.frame(height: 310).clipShape(RoundedRectangle(cornerRadius: 18))
                    HStack {
                        Text("拖动旋转 · 双指缩放").font(.caption).foregroundStyle(.secondary)
                        Spacer(); Button("重置视角") { cameraReset += 1 }.font(.caption)
                    }
                    photoControls
                    DisclosureGroup("相机参数") {
                        VStack(alignment: .leading, spacing: 10) {
                            Picker("定位使用", selection: $model.cameraMode) {
                                Text(CameraMode.auto.title).tag(CameraMode.auto)
                                if model.calibration != nil { Text(CameraMode.saved.title).tag(CameraMode.saved) }
                                if model.selectedRoom?.datasetFormat == "7-scenes" { Text(CameraMode.dataset.title).tag(CameraMode.dataset) }
                            }.pickerStyle(.menu)
                            Button("导入标定 JSON") { importCalibration = true }.disabled(model.selectedRoom == nil || model.busy)
                            Text("自动模式会估算焦距。7-Scenes 参数只适用于原始 640×480 数据集照片。手机照片请使用自动模式或对应镜头的标定。")
                                .font(.caption).foregroundStyle(.secondary)
                        }.padding(.top, 8)
                    }.disabled(model.busy)
                    if model.busy { HStack { ProgressView(); Text(model.message).font(.subheadline) } }
                    else { Text(model.message).font(.subheadline).foregroundStyle(.secondary) }
                    if let error = model.error { Text(error).font(.subheadline).foregroundStyle(.red).textSelection(.enabled) }
                    if let result = model.result { resultCard(result) }
                    Text("GLB 导入、参考库准备与照片定位均在本机完成。使用 RootSIFT + PnP/RANSAC。")
                        .font(.caption).foregroundStyle(.secondary)
                    DisclosureGroup("开源许可") {
                        Text((try? String(contentsOf: Bundle.main.url(forResource: "ThirdPartyNotices", withExtension: "txt") ?? URL(fileURLWithPath: "/missing"), encoding: .utf8)) ?? "许可文件读取失败。")
                            .font(.caption2).textSelection(.enabled)
                    }.font(.caption)
                }.padding()
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("Room Locate")
            .task { await model.start() }
            .onChange(of: model.cameraMode) { _ in model.cameraModeChanged() }
            .onChange(of: selection) { item in
                guard let item else { return }
                Task {
                    do {
                        if let data = try await item.loadTransferable(type: Data.self) { await model.choosePhoto(data) }
                        else { model.error = "未能读取选中的照片。" }
                    } catch { model.error = error.localizedDescription }
                    selection = nil
                }
            }
            .fileImporter(isPresented: $importRoom, allowedContentTypes: [.roomLocalizationPackage]) { result in
                switch result { case .success(let url): Task { await model.importRoom(url) }; case .failure(let error): model.error = error.localizedDescription }
            }
            .fileImporter(isPresented: $importCalibration, allowedContentTypes: [.json]) { result in
                switch result { case .success(let url): Task { await model.importCalibration(url) }; case .failure(let error): model.error = error.localizedDescription }
            }
            .sheet(isPresented: $createRoom) { NewRoomView(model: model) }
            .fullScreenCover(isPresented: $showCamera) {
                CameraCapture { data in showCamera = false; if let data { Task { await model.choosePhoto(data) } } }
                    .ignoresSafeArea()
            }
        }
    }

    private var roomControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("当前房间").font(.headline); Spacer()
                Menu {
                    Button { createRoom = true } label: { Label("GLB 和参考数据", systemImage: "plus") }
                    Button { importRoom = true } label: { Label("已有 .roomloc 房间包", systemImage: "square.and.arrow.down") }
                } label: { Label("导入", systemImage: "square.and.arrow.down") }.disabled(model.busy)
            }
            if model.rooms.isEmpty {
                Text("选择已有 GLB 与参考数据 ZIP，手机会生成并保存定位参考库。").font(.subheadline).foregroundStyle(.secondary)
                Button { createRoom = true } label: { Label("新建房间", systemImage: "plus") }
                    .buttonStyle(.borderedProminent).disabled(model.busy)
            } else {
                Picker("房间", selection: Binding<UUID?>(get: { model.selectedRoom?.id }, set: { id in
                    if let room = model.rooms.first(where: { $0.id == id }) { Task { await model.select(room) } }
                })) {
                    if model.selectedRoom == nil { Text("请选择房间").tag(nil as UUID?) }
                    ForEach(model.rooms) { room in Text(room.name).tag(Optional(room.id)) }
                }.pickerStyle(.menu).disabled(model.busy)
                if let room = model.selectedRoom { Text("\(room.references) 张参考图 · 房间坐标单位：米").font(.caption).foregroundStyle(.secondary) }
            }
        }
    }

    private var photoControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("照片定位").font(.headline)
            if let photo = model.photo {
                Image(uiImage: photo.preview).resizable().scaledToFit().frame(maxHeight: 200)
                    .frame(maxWidth: .infinity).clipShape(RoundedRectangle(cornerRadius: 12))
            }
            HStack {
                Button {
                    Task {
                        if await CameraCapture.requestAccess() { showCamera = true }
                        else { model.error = "相机不可用。请在 iPhone 设置中允许本 App 使用相机，或从相册、文件选择照片。" }
                    }
                } label: { Label("拍照", systemImage: "camera") }
                PhotosPicker(selection: $selection, matching: .images) { Label("相册", systemImage: "photo") }
                Button { importPhoto = true } label: { Label("文件", systemImage: "folder") }
                    .fileImporter(isPresented: $importPhoto, allowedContentTypes: [.image]) { result in
                        switch result {
                        case .success(let url): Task { await model.choosePhotoFile(url) }
                        case .failure(let error): model.error = error.localizedDescription
                        }
                    }
            }.buttonStyle(.bordered).disabled(model.busy || model.selectedRoom == nil)
            Button { Task { await model.localize() } } label: {
                Label("定位这张照片", systemImage: "location.viewfinder").frame(maxWidth: .infinity).padding(.vertical, 6)
            }.buttonStyle(.borderedProminent).tint(Color(red: 0.16, green: 0.36, blue: 0.30))
                .disabled(model.busy || model.photo == nil || model.selectedRoom == nil)
        }
    }

    private func resultCard(_ result: LocalizationResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(result.success ? "已返回拍照位姿" : "暂时无法定位", systemImage: result.success ? "location.fill" : "viewfinder")
                .font(.headline).foregroundStyle(result.success ? .green : .secondary)
            if let position = result.pose?.position_m, position.count == 3 {
                Text(String(format: "X %.3f m   Y %.3f m   Z %.3f m", position[0], position[1], position[2]))
                    .font(.system(.subheadline, design: .monospaced)).textSelection(.enabled)
            }
            Text(String(format: "%d 个几何内点 · %.2f 秒", result.inliers, result.elapsed_seconds)).font(.subheadline)
            if result.camera_estimated {
                Text("相机参数为估算或近似值。返回位姿表示通过几何检查，实际精度仍需独立验证。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let url = model.resultURL { ShareLink(item: url) { Label("导出位姿 JSON", systemImage: "square.and.arrow.up") } }
        }.padding().frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(uiColor: .secondarySystemGroupedBackground)).clipShape(RoundedRectangle(cornerRadius: 16))
    }
}
