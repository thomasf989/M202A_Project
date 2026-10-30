import Foundation
import UniformTypeIdentifiers

extension UTType {
    static let roomLocalizationPackage = UTType(exportedAs: "com.m202a.roomloc", conformingTo: .data)
}

enum RoomLocateError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case let .message(value) = self { return value }; return nil }
}

struct RoomRecord: Codable, Identifiable, Hashable {
    let id: UUID
    let name: String
    let references: Int
    let datasetFormat: String
    let datasetCameraJSON: String
}

struct NativeMetadata: Decodable {
    let name: String
    let references: Int
    let dataset_format: String
    let dataset_camera_json: String
}

struct Calibration: Codable {
    let width: Double
    let height: Double
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    var distortion: [Double]?

    func validate() throws {
        guard [width, height, fx, fy, cx, cy].allSatisfy({ $0.isFinite }),
              width >= 32, height >= 32, width.rounded() == width, height.rounded() == height,
              fx > 0, fy > 0 else { throw RoomLocateError.message("相机参数的尺寸和焦距无效。") }
        if let d = distortion {
            guard [4, 5, 8, 12, 14].contains(d.count), d.allSatisfy({ $0.isFinite }) else {
                throw RoomLocateError.message("畸变参数数量或数值无效。")
            }
        }
    }
}

struct Pose: Decodable {
    let T_world_camera_cv: [[Double]]
    let position_m: [Double]
    let forward_world: [Double]
}

struct LocalizationResult: Decodable {
    let success: Bool
    let reason: String
    let inliers: Int
    let inlier_ratio: Double
    let occupied_grid_cells: Int
    let supporting_references: Int
    let elapsed_seconds: Double
    let camera_estimated: Bool
    let camera_source: String
    let pose: Pose?

    var failureMessage: String {
        switch reason {
        case "no_valid_depth_matches", "too_few_2d_3d_correspondences": return "可用的图像与三维匹配点不足，请换一个清晰、有细节的视角。"
        case "pnp_ransac_failed": return "几何位置计算未通过检查，请重拍或导入对应镜头的标定参数。"
        case "nearly_collinear_3d_points": return "匹配点过于集中，无法可靠估计完整姿态。"
        default: return "匹配点数量或分布未达到定位要求，请换一个视角。"
        }
    }
}

enum CameraMode: String, CaseIterable, Identifiable {
    case auto, saved, dataset
    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "自动估算"
        case .saved: return "已导入标定"
        case .dataset: return "7-Scenes 近似参数"
        }
    }
}
