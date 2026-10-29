import SwiftUI
import ARKit
import RealityKit

@main
struct WalkingApp: App {
    var body: some Scene { WindowGroup { WalkingView() } }
}

struct WalkingView: View {
    @StateObject private var controller = WalkingController()
    var body: some View {
        ZStack(alignment: .bottom) {
            CameraView(controller: controller).ignoresSafeArea()
            VStack(spacing: 12) {
                Text(controller.status).font(.headline)
                Text("Baseline: the blue cube is fixed 50 cm above the printed marker. Walk and turn to check drift.")
                    .font(.caption)
                HStack {
                    Button("Align to marker") { controller.align() }
                        .disabled(!controller.canAlign)
                    Button("Reset session") { controller.reset() }
                }.buttonStyle(.borderedProminent)
            }
            .padding().background(.regularMaterial).cornerRadius(16).padding()
        }
        .onDisappear { controller.pause() }
    }
}

struct CameraView: UIViewRepresentable {
    let controller: WalkingController
    func makeUIView(context: Context) -> ARView {
        controller.start()
        return controller.view
    }
    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// Session delegate callbacks are delivered on the main queue.
final class WalkingController: NSObject, ObservableObject, ARSessionDelegate {
    let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
    @Published var status = "Point the camera at the printed marker."
    @Published var canAlign = false
    private var configuration: ARWorldTrackingConfiguration?
    private var content: AnchorEntity?
    private var trackingNormal = false
    private var interrupted = false

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else {
            status = "This device does not support AR world tracking."
            return
        }
        guard let image = UIImage(named: "alignment-marker"), let cgImage = image.cgImage else {
            status = "Bundled alignment marker is missing."
            return
        }
        let reference = ARReferenceImage(cgImage, orientation: .up, physicalWidth: 0.16)
        reference.name = "room-origin"
        let config = ARWorldTrackingConfiguration()
        config.detectionImages = [reference]
        config.maximumNumberOfTrackedImages = 1
        config.worldAlignment = .gravity
        configuration = config
        view.session.delegate = self
        view.session.delegateQueue = .main
        reset()
    }

    func reset() {
        guard let configuration else { return }
        content?.removeFromParent()
        content = nil
        trackingNormal = false
        canAlign = false
        interrupted = false
        status = "Find the printed marker, then tap Align to marker."
        view.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func pause() { view.session.pause() }

    func align() {
        guard trackingNormal, !interrupted,
              let marker = view.session.currentFrame?.anchors.compactMap({ $0 as? ARImageAnchor })
                .first(where: { $0.referenceImage.name == "room-origin" && $0.isTracked }) else {
            status = "Keep the marker visible with normal camera tracking."
            return
        }
        content?.removeFromParent()
        // Baseline map origin is the printed marker. A later surveyed map can
        // provide mapFromMarker here without changing the rendering pipeline.
        let sessionFromMap = Alignment.sessionFromMap(
            sessionFromMarker: marker.transform, mapFromMarker: matrix_identity_float4x4)
        let root = AnchorEntity(world: sessionFromMap)
        let cube = ModelEntity(mesh: .generateBox(size: 0.08),
                               materials: [SimpleMaterial(color: .systemBlue, isMetallic: false)])
        // ARImageAnchor's local +Y is normal to its image plane. Lay the print flat.
        cube.position = [0, 0.5, 0]
        root.addChild(cube)
        let origin = ModelEntity(mesh: .generateSphere(radius: 0.015),
                                 materials: [SimpleMaterial(color: .systemOrange, isMetallic: false)])
        root.addChild(origin)
        view.scene.addAnchor(root)
        content = root
        status = "Aligned. Walk around; return to the marker to inspect drift."
        // This is a fixed world anchor, not a continuously image-tracked anchor.
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if case .normal = frame.camera.trackingState { trackingNormal = true }
        else { trackingNormal = false }
        canAlign = trackingNormal && !interrupted && frame.anchors.contains {
            guard let marker = $0 as? ARImageAnchor else { return false }
            return marker.referenceImage.name == "room-origin" && marker.isTracked
        }
        content?.isEnabled = trackingNormal && !interrupted
        if !trackingNormal {
            status = "Tracking limited. Move slowly in a well-lit, textured area."
        } else if content != nil {
            status = "Aligned. Walk and turn; tap Align to correct drift when the marker is visible."
        } else {
            status = canAlign ? "Marker found. Tap Align to marker." : "Find the printed marker."
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        interrupted = true
        canAlign = false
        content?.isEnabled = false
        status = "Session interrupted. Return to the app to restart alignment."
    }

    func sessionInterruptionEnded(_ session: ARSession) { reset() }

    func session(_ session: ARSession, didFailWithError error: Error) {
        interrupted = true
        canAlign = false
        content?.isEnabled = false
        status = "Camera session failed: \(error.localizedDescription). Check camera permission, then reset."
    }
}
