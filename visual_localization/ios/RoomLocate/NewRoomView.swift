import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct NewRoomView: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var maxReferences = 120.0
    @State private var choosingModel = true
    @State private var showFiles = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("从已有的模型与参考数据建立房间。解包、提取特征、建立参考库均在这台手机完成。")
                    TextField("房间名称，例如 Fire", text: $name)
                        .textInputAutocapitalization(.words).disabled(model.busy)
                }
                Section("选择文件") {
                    Button {
                        choosingModel = true; showFiles = true
                    } label: {
                        Label(model.glbInput?.name ?? "选择 GLB 模型", systemImage: "cube")
                            .lineLimit(2)
                    }.disabled(model.busy)
                    Button {
                        choosingModel = false; showFiles = true
                    } label: {
                        Label(model.referenceInput?.name ?? "选择参考数据 ZIP", systemImage: "photo.on.rectangle.angled")
                            .lineLimit(2)
                    }.disabled(model.busy)
                    Text("可从“我的 iPhone”或 iCloud Drive 选择。云端文件需要先下载；文件准备好后，建库和定位均无需联网。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("参考数据要求") {
                    Text("支持 7-Scenes 原始或已准备数据、Polycam 校正后的原始数据，以及含 references.json 的照片与位姿数据。")
                    Text("至少需要 4 张有效参考照片，并包含相机参数、拍摄位姿。模型与参考数据须使用对应坐标和米制单位。测试组照片不会参与建库。")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                Section("建库规模") {
                    Text("最多使用 \(Int(maxReferences)) 张参考照片")
                    Slider(value: $maxReferences, in: 20...200, step: 20).disabled(model.busy)
                    Text("超出上限时均匀抽取参考帧。增加照片能扩大覆盖范围，也会增加耗时和内存。建库时请保持 App 在前台。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if model.busy {
                    Section {
                        if model.building {
                            ProgressView(value: model.buildProgress)
                            Text("\(Int(model.buildProgress * 100))% · \(model.message)").font(.subheadline)
                            Button("取消准备", role: .cancel) { model.cancelBuild() }
                        } else { HStack { ProgressView(); Text(model.message).font(.subheadline) } }
                    }
                }
                if let error = model.error {
                    Section { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                }
                Section {
                    Button {
                        Task { if await model.buildRoom(name: name, maxReferences: Int(maxReferences)) { dismiss() } }
                    } label: {
                        Label("在手机上准备房间", systemImage: "iphone").frame(maxWidth: .infinity)
                    }.disabled(model.busy || model.glbInput == nil || model.referenceInput == nil || name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .navigationTitle("新建房间").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("关闭") { dismiss() }.disabled(model.busy) } }
            .interactiveDismissDisabled(model.busy)
            .onAppear { suggestName() }
            .onChange(of: model.glbInput?.url) { _ in suggestName() }
            .fileImporter(isPresented: $showFiles, allowedContentTypes: choosingModel ? [.roomGLBModel] : [.zip]) { result in
                switch result {
                case .success(let url):
                    let isModel = choosingModel
                    Task { await model.stageInput(url, isModel: isModel) }
                case .failure(let error): model.error = error.localizedDescription
                }
            }
        }
    }

    private func suggestName() {
        if name.isEmpty, let file = model.glbInput {
            name = (file.name as NSString).deletingPathExtension
        }
    }
}
