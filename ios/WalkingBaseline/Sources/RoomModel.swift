import Foundation
import simd

struct WiFiSample: Codable {
    var position: [Float]
    var dbm: Float
    var measuredAt: String
}
struct WiFiSurvey: Codable {
    var mapID: String
    var network: String
    var radius: Float
    var samples: [WiFiSample]
    func validated(for map: String) throws -> WiFiSurvey {
        guard mapID == map, !network.isEmpty, radius.isFinite, radius > 0, radius <= 5,
              samples.count <= 5000, samples.allSatisfy({
                  $0.position.count == 3 && $0.position.allSatisfy(\.isFinite) &&
                  $0.dbm.isFinite && (-120 ... -10).contains($0.dbm) && !$0.measuredAt.isEmpty
              }) else { throw RoomError.invalidSurvey }
        return self
    }
    // 3D inverse-distance estimate, bounded by survey radius. No global extrapolation.
    func estimate(at p: SIMD3<Float>) -> Float? {
        let near = samples.map { (simd_distance(p, SIMD3($0.position[0], $0.position[1], $0.position[2])), $0.dbm) }
            .filter { $0.0 <= radius }.sorted { $0.0 < $1.0 }.prefix(8)
        guard let first = near.first else { return nil }
        if first.0 < 0.01 { return first.1 }
        var total: Float = 0; var weights: Float = 0
        for (d, value) in near { let w = 1 / max(d*d, 0.0001); total += value*w; weights += w }
        return total / weights
    }
}
enum RoomError: LocalizedError {
    case invalidRoom, invalidSurvey
    var errorDescription: String? {
        switch self {
        case .invalidRoom: return "Room package is missing or invalid. Run the room preparation script on your Mac, then rebuild."
        case .invalidSurvey: return "Survey must match this map and contain valid positions, dBm values, dates, and a radius up to 5 m."
        }
    }
}
struct RoomModel: Decodable {
    let mapID: String
    let name: String
    let vertices: [[Float]]
    let triangles: [[Int]]
    var points: [SIMD3<Float>] { vertices.map { SIMD3($0[0], $0[1], $0[2]) } }
    static func load(from folder: URL) throws -> RoomModel {
        let url = folder.appendingPathComponent("room.json")
        let r = try JSONDecoder().decode(RoomModel.self, from: Data(contentsOf: url))
        guard !r.mapID.isEmpty, !r.vertices.isEmpty, r.vertices.count <= 500_000,
              r.vertices.allSatisfy({$0.count == 3 && $0.allSatisfy(\.isFinite)}),
              r.triangles.count <= 1_000_000, r.triangles.allSatisfy({$0.count == 3 && $0.allSatisfy({r.vertices.indices.contains($0)})}) else { throw RoomError.invalidRoom }
        return r
    }
    // Nearest double-sided triangle hit, in the GLB map frame. No arbitrary depth plane.
    func hit(origin: SIMD3<Float>, direction: SIMD3<Float>) -> SIMD3<Float>? {
        let p = points
        var nearest: Float = 30
        var found = false
        for t in triangles {
            let a=p[t[0]], e1=p[t[1]]-a, e2=p[t[2]]-a
            let h=simd_cross(direction,e2), det=simd_dot(e1,h)
            if abs(det)<1e-7 { continue }
            let s=origin-a, u=simd_dot(s,h)/det
            if u<0 || u>1 { continue }
            let q=simd_cross(s,e1), v=simd_dot(direction,q)/det
            if v<0 || u+v>1 { continue }
            let d=simd_dot(e2,q)/det
            if d>0.05 && d<nearest { nearest=d; found=true }
        }
        return found ? origin+nearest*direction : nil
    }
}
extension simd_float4x4 {
    var position: SIMD3<Float> { SIMD3(columns.3.x, columns.3.y, columns.3.z) }
    static func fromRows(_ values: [NSNumber]) -> simd_float4x4? {
        guard values.count==16, values.allSatisfy({$0.floatValue.isFinite}) else { return nil }
        var m=matrix_identity_float4x4
        for r in 0..<4 { for c in 0..<4 { m[c][r]=values[r*4+c].floatValue } }
        let rot=simd_float3x3(SIMD3(m[0].x,m[0].y,m[0].z), SIMD3(m[1].x,m[1].y,m[1].z), SIMD3(m[2].x,m[2].y,m[2].z))
        guard abs(simd_determinant(rot)-1)<0.01, abs(m[0].w)+abs(m[1].w)+abs(m[2].w)+abs(m[3].w-1)<0.001 else { return nil }
        let check=simd_transpose(rot)*rot
        for c in 0..<3 { if simd_length(check[c]-matrix_identity_float3x3[c])>0.01 { return nil } }
        return m
    }
}
