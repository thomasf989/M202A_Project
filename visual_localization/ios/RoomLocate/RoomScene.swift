import Foundation
import SceneKit
import SwiftUI
import simd

struct RoomMesh {
    let data: Data
    let vertexCount: Int
    let faceCount: Int
    let center: SCNVector3
    let span: Float

    static func validate(at url: URL) throws { _ = try load(at: url) }

    static func load(at url: URL) throws -> RoomMesh {
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size >= 16, size <= 64 * 1024 * 1024 else { throw RoomLocateError.message("房间显示模型大小无效。") }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard String(data: data.prefix(8), encoding: .ascii) == "RMSH0001" else {
            throw RoomLocateError.message("房间显示模型格式无效。")
        }
        func u32(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        let nv = Int(u32(8)), nf = Int(u32(12))
        guard nv > 0, nv <= 200_000, nf > 0, nf <= 300_000,
              size == 16 + nv * 40 + nf * 12 else { throw RoomLocateError.message("房间网格数量或长度无效。") }
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for i in 0..<nv {
            var values = [Float](); values.reserveCapacity(10)
            for j in 0..<10 {
                let value = Float(bitPattern: u32(16 + i * 40 + j * 4))
                guard value.isFinite, abs(value) < 1e6 else { throw RoomLocateError.message("模型中存在无效坐标。") }
                values.append(value)
            }
            let point = SIMD3(values[0], values[1], values[2])
            minimum = simd_min(minimum, point); maximum = simd_max(maximum, point)
        }
        for i in 0..<(nf * 3) {
            guard u32(16 + nv * 40 + i * 4) < UInt32(nv) else { throw RoomLocateError.message("模型三角面索引无效。") }
        }
        let c = (minimum + maximum) * 0.5
        return RoomMesh(data: data, vertexCount: nv, faceCount: nf, center: SCNVector3(c.x, c.y, c.z),
                        span: max(simd_length(maximum - minimum), 0.5))
    }

    func node() -> SCNNode {
        let vertices = data.subdata(in: 16..<(16 + vertexCount * 40))
        let source: (SCNGeometrySource.Semantic, Int, Int) -> SCNGeometrySource = { semantic, components, offset in
            SCNGeometrySource(data: vertices, semantic: semantic, vectorCount: vertexCount,
                              usesFloatComponents: true, componentsPerVector: components,
                              bytesPerComponent: 4, dataOffset: offset, dataStride: 40)
        }
        let indices = data.subdata(in: (16 + vertexCount * 40)..<data.count)
        let element = SCNGeometryElement(data: indices, primitiveType: .triangles,
                                        primitiveCount: faceCount, bytesPerIndex: 4)
        let geometry = SCNGeometry(sources: [source(.vertex, 3, 0), source(.normal, 3, 12), source(.color, 4, 24)], elements: [element])
        let material = SCNMaterial(); material.diffuse.contents = UIColor.white
        material.isDoubleSided = true; material.lightingModel = .lambert
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }
}

struct RoomSceneView: UIViewRepresentable {
    let roomID: UUID?
    let mesh: RoomMesh?
    let pose: Pose?
    let reset: Int

    final class Coordinator {
        var roomID: UUID?
        var transform: [[Double]]?
        var reset = -1
    }
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(); view.scene = SCNScene(); view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true; view.backgroundColor = .secondarySystemGroupedBackground
        view.antialiasingMode = .multisampling4X
        return view
    }
    func updateUIView(_ view: SCNView, context: Context) {
        let coordinator = context.coordinator
        if coordinator.roomID != roomID {
            coordinator.roomID = roomID; coordinator.transform = nil; coordinator.reset = -1
            let scene = SCNScene(); if let mesh { scene.rootNode.addChildNode(mesh.node()) }; view.scene = scene
        }
        if let mesh, coordinator.reset != reset {
            coordinator.reset = reset
            let camera = SCNNode(); camera.camera = SCNCamera()
            camera.camera?.zNear = 0.01; camera.camera?.zFar = Double(max(mesh.span * 100, 1000))
            camera.position = SCNVector3(mesh.center.x + mesh.span * 0.8, mesh.center.y + mesh.span * 0.55, mesh.center.z + mesh.span * 0.8)
            camera.look(at: mesh.center)
            view.scene?.rootNode.childNode(withName: "viewerCamera", recursively: false)?.removeFromParentNode()
            camera.name = "viewerCamera"; view.scene?.rootNode.addChildNode(camera); view.pointOfView = camera
            view.defaultCameraController.target = mesh.center
        }
        let transform = pose?.T_world_camera_cv
        if coordinator.transform != transform {
            coordinator.transform = transform
            view.scene?.rootNode.childNode(withName: "localizedCamera", recursively: false)?.removeFromParentNode()
            if let pose, pose.T_world_camera_cv.count == 4, pose.T_world_camera_cv.allSatisfy({ $0.count == 4 }) {
                let node = cameraMarker(pose); node.name = "localizedCamera"; view.scene?.rootNode.addChildNode(node)
            }
        }
    }

    private func cameraMarker(_ pose: Pose) -> SCNNode {
        let node = SCNNode(), t = pose.T_world_camera_cv
        node.simdTransform = simd_float4x4(columns: (
            SIMD4(Float(t[0][0]), Float(t[1][0]), Float(t[2][0]), Float(t[3][0])),
            SIMD4(Float(t[0][1]), Float(t[1][1]), Float(t[2][1]), Float(t[3][1])),
            SIMD4(Float(t[0][2]), Float(t[1][2]), Float(t[2][2]), Float(t[3][2])),
            SIMD4(Float(t[0][3]), Float(t[1][3]), Float(t[2][3]), Float(t[3][3]))))
        let red = SCNMaterial(); red.diffuse.contents = UIColor.systemRed; red.lightingModel = .constant
        let sphere = SCNSphere(radius: 0.045); sphere.materials = [red]; node.addChildNode(SCNNode(geometry: sphere))
        let corners: [SCNVector3] = [SCNVector3(-0.13, -0.10, 0.35), SCNVector3(0.13, -0.10, 0.35),
                                     SCNVector3(0.13, 0.10, 0.35), SCNVector3(-0.13, 0.10, 0.35)]
        var points = [SCNVector3]()
        for i in 0..<4 { points += [SCNVector3Zero, corners[i], corners[i], corners[(i + 1) % 4]] }
        points += [SCNVector3Zero, SCNVector3(0, 0, 0.5)]
        let element = SCNGeometryElement(indices: Array(0..<Int32(points.count)), primitiveType: .line)
        let geometry = SCNGeometry(sources: [SCNGeometrySource(vertices: points)], elements: [element]); geometry.materials = [red]
        node.addChildNode(SCNNode(geometry: geometry)); return node
    }
}
