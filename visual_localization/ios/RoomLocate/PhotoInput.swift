import AVFoundation
import CryptoKit
import ImageIO
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct PreparedPhoto {
    let jpeg: Data
    let preview: UIImage
    let originalWidth: Int
    let originalHeight: Int
    let focal35mm: Double?
    let sourceHash: String

    static func prepare(fileAt url: URL) throws -> PreparedPhoto {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let coordinator = NSFileCoordinator()
        var coordinationError: NSError?, readError: Error?
        var photo: PreparedPhoto?
        coordinator.coordinate(readingItemAt: url, options: [], error: &coordinationError) { readable in
            do {
                let values = try readable.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard values.isRegularFile == true, let size = values.fileSize,
                      size > 0, size <= 128 * 1024 * 1024 else {
                    throw RoomLocateError.message("请选择完整的照片文件，大小不能超过 128 MiB。")
                }
                // Keep original bytes for EXIF orientation, focal metadata and
                // reference-photo hash exclusion, just like the album path.
                photo = try prepare(Data(contentsOf: readable, options: .mappedIfSafe))
            } catch { readError = error }
        }
        if let coordinationError { throw coordinationError }
        if let readError { throw readError }
        guard let photo else { throw RoomLocateError.message("未能读取照片，请在“文件”App 中下载完成后重试。") }
        return photo
    }

    static func prepare(_ data: Data) throws -> PreparedPhoto {
        guard data.count <= 128 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
              let pw = properties[kCGImagePropertyPixelWidth as String] as? NSNumber,
              let ph = properties[kCGImagePropertyPixelHeight as String] as? NSNumber,
              pw.doubleValue * ph.doubleValue <= 65_000_000,
              let image = UIImage(data: data), let cg = image.cgImage else {
            throw RoomLocateError.message("无法读取照片，或照片超过 6500 万像素。")
        }
        let rotated = [UIImage.Orientation.left, .right, .leftMirrored, .rightMirrored].contains(image.imageOrientation)
        let width = rotated ? cg.height : cg.width, height = rotated ? cg.width : cg.height
        guard min(width, height) >= 32 else { throw RoomLocateError.message("照片尺寸太小。") }
        let scale = min(1.0, 1280.0 / Double(max(width, height)))
        let size = CGSize(width: (Double(width) * scale).rounded(), height: (Double(height) * scale).rounded())
        let format = UIGraphicsImageRendererFormat(); format.scale = 1; format.opaque = true
        let oriented = UIGraphicsImageRenderer(size: size, format: format).image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        guard let jpeg = oriented.jpegData(compressionQuality: 0.95) else { throw RoomLocateError.message("无法准备照片。") }
        let exif = properties[kCGImagePropertyExifDictionary as String] as? [String: Any]
        let focal = (exif?[kCGImagePropertyExifFocalLenIn35mmFilm as String] as? NSNumber)?.doubleValue
        return PreparedPhoto(jpeg: jpeg, preview: oriented, originalWidth: width, originalHeight: height,
                             focal35mm: focal, sourceHash: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }

    func cameraJSON(mode: CameraMode, calibration: Calibration?) throws -> String {
        var value: [String: Any] = ["mode": mode.rawValue, "original_width": originalWidth,
                                    "original_height": originalHeight, "source_sha256": sourceHash]
        if let focal35mm, focal35mm.isFinite { value["focal_35mm"] = focal35mm }
        if let calibration { value["calibration"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(calibration)) }
        return String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
    }
}

struct CameraCapture: UIViewControllerRepresentable {
    let completion: (Data?) -> Void
    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let completion: (Data?) -> Void
        init(_ completion: @escaping (Data?) -> Void) { self.completion = completion }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            guard let image = info[.originalImage] as? UIImage, let cg = image.cgImage else { completion(nil); return }
            // Preserve original EXIF and sensor pixel orientation so automatic
            // focal estimation and imported calibration receive consistent data.
            let output = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(output as CFMutableData, UTType.jpeg.identifier as CFString, 1, nil) else {
                completion(nil); return
            }
            var metadata = info[.mediaMetadata] as? [String: Any] ?? [:]
            let exifOrientation: Int
            switch image.imageOrientation {
            case .up: exifOrientation = 1
            case .down: exifOrientation = 3
            case .left: exifOrientation = 8
            case .right: exifOrientation = 6
            case .upMirrored: exifOrientation = 2
            case .downMirrored: exifOrientation = 4
            case .leftMirrored: exifOrientation = 5
            case .rightMirrored: exifOrientation = 7
            @unknown default: exifOrientation = 1
            }
            metadata[kCGImagePropertyOrientation as String] = exifOrientation
            metadata[kCGImageDestinationLossyCompressionQuality as String] = 0.98
            CGImageDestinationAddImage(destination, cg, metadata as CFDictionary)
            completion(CGImageDestinationFinalize(destination) ? output as Data : nil)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { completion(nil) }
    }
    func makeCoordinator() -> Coordinator { Coordinator(completion) }
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController(); picker.sourceType = .camera; picker.cameraDevice = .rear
        picker.delegate = context.coordinator; picker.cameraCaptureMode = .photo; return picker
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    static func requestAccess() async -> Bool {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else { return false }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }
}
