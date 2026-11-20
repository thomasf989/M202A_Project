import SwiftUI
import RealityKit
import UniformTypeIdentifiers

@main
struct WalkingApp: App {
    var body: some SwiftUI.Scene { WindowGroup { ExplorerView() } }
}
struct ExplorerView: View {
    @StateObject private var controller=ExplorerController()
    @Environment(\.scenePhase) private var scenePhase
    @State private var showingTools=false
    @State private var importing=false
    @State private var dbm=""
    @State private var network=""
    @State private var exportURL: URL?
    var body: some View {
        ZStack {
            CameraView(controller:controller).ignoresSafeArea()
            VStack(spacing:12) {
                HStack {
                    VStack(alignment:.leading) {
                        Text("Room Wi-Fi Explorer").font(.headline)
                        Text("OFFLINE · \(controller.roomName)").font(.caption)
                    }
                    Spacer()
                    Button {showingTools=true} label:{Image(systemName:"slider.horizontal.3").font(.title2)}
                        .accessibilityLabel("Survey tools")
                }.padding().background(.regularMaterial,in:RoundedRectangle(cornerRadius:18))
                Spacer()
                VStack(alignment:.leading,spacing:10) {
                    Text(controller.status).font(.subheadline)
                    Text(controller.reading).font(.title3.bold())
                    Text(controller.poseText).font(.caption.monospaced())
                    Text(controller.selected).font(.caption)
                    if controller.sampleCount==0 {
                        Text("No survey yet. Import measurements to see the heatmap.").font(.caption).foregroundStyle(.secondary)
                    } else {
                        HStack {
                            Text("−90 dBm").foregroundStyle(.red)
                            LinearGradient(colors:[.red,.yellow,.green],startPoint:.leading,endPoint:.trailing).frame(height:8)
                            Text("−40 dBm").foregroundStyle(.green)
                        }.font(.caption)
                        Text("\(controller.sampleCount) readings · 3D interpolation · uncovered areas stay blank").font(.caption2)
                    }
                    HStack {
                        Button(action:controller.locate) {
                            HStack {if controller.busy {ProgressView()};Text(controller.aligned ? "Relocate" : "Locate me")}
                        }.disabled(!controller.ready || controller.busy || !controller.supportsAR)
                        Button("Reset",action:controller.reset)
                    }.buttonStyle(.borderedProminent)
                }.padding().background(.regularMaterial,in:RoundedRectangle(cornerRadius:18))
            }.padding()
        }
        .onChange(of:scenePhase) {phase in
            if phase == .active {controller.reset()} else if phase == .background {controller.suspend()}
        }
        .sheet(isPresented:$showingTools) {
            NavigationStack {
                Form {
                    Section("Offline room library") {
                        ForEach(controller.rooms) {room in
                            Button(room.name) {controller.selectRoom(room)}.disabled(controller.busy)
                        }
                        Button("Import room package (.wifiroom)") {importing=true}
                        Text("Prepare each scan once on a computer, then import its package. Localization and walking work offline on this phone.")
                    }
                    Section("Measured Wi-Fi survey") {
                        Text("Import a survey registered to this room’s GLB coordinates, in meters. Values shown are estimates from those measurements, not live iPhone RSSI.")
                        Button("Import survey JSON") {importing=true}
                        Button("Prepare survey export") {exportURL=controller.exportSurvey()}
                        if let exportURL {ShareLink("Share survey JSON",item:exportURL)}
                    }
                    Section("Record an external measurement here") {
                        Text("Locate first. Enter the signal strength measured by your Wi-Fi survey equipment at the current phone position.")
                        TextField("Network / BSSID and band",text:$network)
                        TextField("RSSI in dBm (for example −60)",text:$dbm).keyboardType(.numbersAndPunctuation)
                        Button("Save reading at my location") {
                            if let value=Float(dbm) {controller.addMeasurement(dbm:value,network:network)}
                        }.disabled(!controller.aligned || Float(dbm)==nil)
                    }
                    Section("Status") {Text(controller.status)}
                    Section("About") {
                        Text("All photo localization and AR rendering run on this iPhone. This build uses the project’s RootSIFT/PnP baseline. It does not run the desktop MeshLoc neural pipeline.")
                        Text("OpenCV is distributed under Apache 2.0. See the bundled OpenCV license.")
                    }
                }.navigationTitle("Survey tools").toolbar {Button("Done"){showingTools=false}}
            }
        }
        .fileImporter(isPresented:$importing,allowedContentTypes:[.json, .data]) {result in
            if case let .success(url)=result {
                if url.pathExtension.lowercased()=="wifiroom" {controller.importRoom(url)}
                else {controller.importSurvey(url)}
            }
        }
    }
}
struct CameraView: UIViewRepresentable {
    let controller: ExplorerController
    func makeUIView(context:Context) -> ARView {controller.start();return controller.view}
    func updateUIView(_ uiView:ARView,context:Context) {}
}
