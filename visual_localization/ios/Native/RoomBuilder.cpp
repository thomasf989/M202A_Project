#include "RoomBuilder.hpp"
#include "LocalizerCore.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#define CGLTF_IMPLEMENTATION
#include "vendor/cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 16384
#include "vendor/stb_image.h"
#include "vendor/picosha2.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace roomloc {
namespace {
namespace fs = std::filesystem;
using Vec = cv::Vec3f;
struct Mesh {
    std::vector<Vec> vertices;
    std::vector<cv::Vec3i> faces;
    std::vector<cv::Vec4f> colors;
};
struct Frame {
    std::string id;
    fs::path image;
    cv::Matx44d pose;
    cv::Matx33d K;
    cv::Size size;
};
struct Capture { std::vector<Frame> frames; std::string format, camera = "{}"; };
struct Reference { std::string id, hash; cv::Mat descriptors, world, valid; };

void report(const BuildProgress& progress, double value, const std::string& text) {
    if (progress && !progress(value, text)) throw std::runtime_error("已取消房间准备。");
}
void checkpoint(const BuildProgress& progress) { report(progress, -1, ""); }
std::vector<unsigned char> readBytes(const fs::path& path, uint64_t limit) {
    if (!fs::is_regular_file(path) || fs::file_size(path) > limit) throw std::runtime_error("文件缺失或超出手机大小限制：" + path.filename().string());
    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> data(size_t(fs::file_size(path)));
    if (!input.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()))) throw std::runtime_error("文件读取失败：" + path.filename().string());
    return data;
}
cv::FileStorage json(const fs::path& path) {
    auto bytes = readBytes(path, 24 * 1024 * 1024);
    cv::FileStorage result(std::string(bytes.begin(), bytes.end()), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    if (!result.isOpened()) throw std::runtime_error("无法读取参考数据 JSON。");
    return result;
}
double number(const cv::FileNode& value) {
    if ((!value.isInt() && !value.isReal()) || !std::isfinite(double(value))) throw std::runtime_error("参考数据包含缺失或无效数值。");
    return double(value);
}
cv::Matx44d matrix(const cv::FileNode& node, bool optional = false) {
    if (node.empty() && optional) return cv::Matx44d::eye();
    if (!node.isSeq() || node.size() != 4) throw std::runtime_error("参考位姿必须为 4×4 矩阵。");
    cv::Matx44d result;
    for (int i = 0; i < 4; ++i) {
        if (!node[i].isSeq() || node[i].size() != 4) throw std::runtime_error("参考位姿必须为 4×4 矩阵。");
        for (int j = 0; j < 4; ++j) result(i, j) = number(node[i][j]);
    }
    return result;
}
cv::Matx44d rigid(cv::Matx44d value, bool kinect = false) {
    cv::Matx33d R;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) R(i, j) = value(i, j);
    if (!cv::checkRange(cv::Mat(value)) || cv::norm(cv::Mat(value).row(3), cv::Mat(cv::Matx14d(0, 0, 0, 1)), cv::NORM_INF) > 1e-5 ||
        cv::norm(cv::Mat(R.t() * R - cv::Matx33d::eye()), cv::NORM_INF) > (kinect ? .001 : .0001) || std::abs(cv::determinant(R) - 1) > (kinect ? .002 : .0001))
        throw std::runtime_error("相机位姿包含无效旋转、缩放或反射。");
    if (kinect) {
        cv::SVD svd(cv::Mat(R), cv::SVD::FULL_UV); cv::Mat corrected = svd.u * svd.vt;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) value(i, j) = corrected.at<double>(i, j);
    }
    return value;
}
Frame frame(const fs::path& root, const std::string& id, const std::string& relative, const cv::FileNode& camera, const cv::Matx44d& pose) {
    static const std::regex identifier("[A-Za-z0-9_-]{1,100}");
    fs::path path(relative);
    if (!std::regex_match(id, identifier) || path.empty() || path.is_absolute() || relative.find('\\') != std::string::npos)
        throw std::runtime_error("参考照片编号或路径无效。");
    for (const auto& part : path) if (part == ".." || part == ".") throw std::runtime_error("参考照片路径不能离开参考数据目录。");
    auto resolved = fs::weakly_canonical(root / path).lexically_relative(fs::canonical(root));
    if (resolved.empty() || *resolved.begin() == "..") throw std::runtime_error("参考照片路径不安全。");
    double w = number(camera["width"]), h = number(camera["height"]);
    double fx = number(camera["fx"]), fy = number(camera["fy"]), cx = number(camera["cx"]), cy = number(camera["cy"]);
    if (w < 32 || h < 32 || w > 16384 || h > 16384 || std::floor(w) != w || std::floor(h) != h || w * h > 24e6 || fx <= 0 || fy <= 0 ||
        fx > 1e7 || fy > 1e7 || std::abs(cx) > w * 16 || std::abs(cy) > h * 16)
        throw std::runtime_error("参考照片的相机参数或尺寸无效。");
    for (const auto& d : camera["distortion"]) if (number(d) != 0) throw std::runtime_error("参考照片需要先去除镜头畸变，并提供对应的相机参数。");
    return {id, root / path, rigid(pose), cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1), cv::Size(int(w), int(h))};
}
cv::Matx44d flip() { return cv::Matx44d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1); }
std::set<std::string> split(const fs::path& path) {
    auto bytes = readBytes(path, 65536); std::istringstream stream(std::string(bytes.begin(), bytes.end()));
    std::string line; std::set<std::string> result;
    while (stream >> line) {
        if (!std::regex_match(line, std::regex("sequence[0-9]{1,4}"))) throw std::runtime_error("7-Scenes 官方划分文件无效。");
        std::ostringstream seq; seq << "seq-" << std::setfill('0') << std::setw(2) << std::stoi(line.substr(8));
        if (!result.insert(seq.str()).second) throw std::runtime_error("7-Scenes 划分文件包含重复序列。");
    }
    if (result.empty()) throw std::runtime_error("7-Scenes 官方划分文件不能为空。");
    return result;
}
void disjoint(const std::set<std::string>& a, const std::set<std::string>& b) {
    for (const auto& id : a) if (b.count(id)) throw std::runtime_error("训练组和测试组发生重叠，已停止建库。");
}
cv::Matx44d poseFile(const fs::path& path) {
    auto bytes = readBytes(path, 8192); std::istringstream stream(std::string(bytes.begin(), bytes.end())); cv::Matx44d T;
    for (double& v : T.val) {
        std::string token; if (!(stream >> token)) throw std::runtime_error("7-Scenes 位姿文件不完整。");
        size_t end = 0; v = std::stod(token, &end);
        if (end != token.size() || !std::isfinite(v)) throw std::domain_error("Invalid KinectFusion frame");
    }
    std::string extra; if (stream >> extra) throw std::runtime_error("7-Scenes 位姿文件长度无效。");
    return rigid(T, true);
}
std::string cameraJSON(const cv::FileNode& c) {
    cv::FileStorage storage("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    for (const char* key : {"width", "height", "fx", "fy", "cx", "cy"}) storage << key << number(c[key]);
    return storage.releaseAndGetString();
}

Capture readCapture(const fs::path& source, int maximum, const BuildProgress& progress) {
    std::set<fs::path> roots; size_t visited = 0;
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
        if (++visited > 100000 || entry.is_symlink()) throw std::runtime_error("参考数据包含符号链接或文件数量过多。");
        if (visited % 256 == 0) checkpoint(progress);
        const auto name = entry.path().filename();
        if (name == "references.json" || name == "seven_scenes.json" || name == "TrainSplit.txt" || name == "optimized_roomplan.json") roots.insert(entry.path().parent_path());
    }
    if (roots.size() != 1) throw std::runtime_error("请选择单个房间的参考 ZIP：支持 7-Scenes、Polycam 原始数据，或含 references.json 的照片与位姿数据。");
    fs::path root = *roots.begin(); Capture capture; std::set<std::string> seen;
    if (fs::exists(root / "references.json")) {
        auto meta = json(root / "references.json"); auto m = meta.root();
        if (std::string(m["format"]) != "posed-images-v1") throw std::runtime_error("references.json 的 format 必须为 posed-images-v1。");
        if (!m["units"].empty() && std::string(m["units"]) != "meters") throw std::runtime_error("参考位姿和 GLB 必须使用米制坐标。");
        auto A = rigid(matrix(m["world_to_glb"], true));
        std::string convention = m["camera_convention"].empty() ? "opencv" : std::string(m["camera_convention"]);
        if (convention != "opencv" && convention != "opengl") throw std::runtime_error("未知的相机坐标约定。");
        auto C = convention == "opengl" ? flip() : cv::Matx44d::eye();
        if (!m["frames"].isSeq() || m["frames"].size() > 20000) throw std::runtime_error("参考照片列表无效或过大。");
        for (const auto& r : m["frames"]) {
            if (seen.size() % 128 == 0) checkpoint(progress);
            std::string id = r["id"].empty() ? "photo-" + std::to_string(seen.size()) : std::string(r["id"]);
            if (!seen.insert(id).second) throw std::runtime_error("参考数据包含重复照片编号。");
            std::string group = r["split"].empty() ? "reference" : std::string(r["split"]);
            if (group == "test") continue;
            if (group != "reference" && group != "train") throw std::runtime_error("照片划分必须是 reference、train 或 test。");
            capture.frames.push_back(frame(root, id, std::string(r["image"]), r["intrinsics"].empty() ? m["intrinsics"] : r["intrinsics"], A * rigid(matrix(r["camera_to_world"])) * C));
        }
        capture.format = "posed-images";
    } else if (fs::exists(root / "seven_scenes.json")) {
        auto meta = json(root / "seven_scenes.json"); auto m = meta.root();
        if (std::string(m["format"]) != "7-scenes-v1" || std::string(m["depth_source"]) != "mesh_rendered_rgb") throw std::runtime_error("不支持此 7-Scenes 清单格式。");
        auto A = rigid(matrix(m["world_to_glb"]));
        std::set<std::string> train, test;
        for (const auto& value : m["train_sequences"]) train.insert(std::string(value));
        for (const auto& value : m["test_sequences"]) test.insert(std::string(value));
        disjoint(train, test);
        if (!m["frames"].isSeq() || m["frames"].size() > 20000) throw std::runtime_error("7-Scenes 帧列表无效或过大。");
        for (const auto& r : m["frames"]) {
            if (seen.size() % 128 == 0) checkpoint(progress);
            std::string seq = r["sequence"], stem = r["stem"], id = r["id"], group = r["split"];
            if (!std::regex_match(seq, std::regex("seq-[0-9]+")) || !std::regex_match(stem, std::regex("frame-[0-9]+")) ||
                id != seq + "_" + stem || !seen.insert(id).second ||
                (group != "train" && group != "test") || !(group == "train" ? train : test).count(seq)) throw std::runtime_error("7-Scenes 帧编号或训练测试划分无效。");
            if (group == "test") continue;
            try { capture.frames.push_back(frame(root, id, seq + "/" + stem + ".color.png", m["rgb_intrinsics"], A * poseFile(root / seq / (stem + ".pose.txt")))); }
            catch (const std::domain_error&) { /* Official invalid tracking frames are not references. */ }
        }
        capture.format = "7-scenes"; capture.camera = cameraJSON(m["rgb_intrinsics"]);
    } else if (fs::exists(root / "TrainSplit.txt")) {
        auto train = split(root / "TrainSplit.txt"), test = split(root / "TestSplit.txt"); disjoint(train, test);
        auto profile = cv::FileStorage("{\"width\":640,\"height\":480,\"fx\":525.0,\"fy\":525.0,\"cx\":320.0,\"cy\":240.0}", cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        for (const auto& seq : train) {
            // The test sequence directory and its images/poses are never opened.
            if (!fs::is_directory(root / seq)) continue;
            std::vector<fs::path> images;
            for (const auto& e : fs::directory_iterator(root / seq)) if (std::regex_match(e.path().filename().string(), std::regex("frame-[0-9]+\\.color\\.png"))) images.push_back(e.path());
            std::sort(images.begin(), images.end());
            for (const auto& image : images) {
                std::string stem = image.filename().string(); stem.resize(stem.size() - 10);
                // Validate only the uniformly sampled subset of large raw datasets below.
                capture.frames.push_back({seq + "_" + stem, image, cv::Matx44d::eye(), cv::Matx33d(525, 0, 320, 0, 525, 240, 0, 0, 1), {640, 480}});
            }
        }
        capture.format = "7-scenes"; capture.camera = cameraJSON(profile.root());
    } else {
        auto meta = json(root / "optimized_roomplan.json"); auto a = meta["alignment_transform"]; cv::Matx44d A;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) A(i, j) = number(a["t" + std::to_string(i) + std::to_string(j)]);
        rigid(A); std::vector<fs::path> cameras;
        for (const auto& e : fs::directory_iterator(root / "keyframes/corrected_cameras")) if (e.path().extension() == ".json") cameras.push_back(e.path());
        std::sort(cameras.begin(), cameras.end());
        for (const auto& file : cameras) {
            if (capture.frames.size() % 128 == 0) checkpoint(progress);
            auto c = json(file); cv::Matx44d T = cv::Matx44d::eye();
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j) T(i, j) = number(c["t_" + std::to_string(i) + std::to_string(j)]);
            capture.frames.push_back(frame(root, file.stem().string(), "keyframes/corrected_images/" + file.stem().string() + ".jpg", c.root(), A * rigid(T) * flip()));
        }
        capture.format = "polycam-mesh";
    }
    if (capture.frames.size() > 20000) throw std::runtime_error("参考照片列表超过手机限制。");
    std::vector<Frame> selected;
    size_t count = std::min(capture.frames.size(), size_t(maximum));
    for (size_t i = 0; i < count; ++i) {
        auto f = capture.frames[i * capture.frames.size() / count];
        if (capture.format == "7-scenes" && !fs::exists(root / "seven_scenes.json")) {
            auto pose = f.image; std::string name = pose.filename().string(); name.resize(name.size() - 10);
            try { f.pose = flip() * poseFile(pose.parent_path() / (name + ".pose.txt")); }
            catch (const std::domain_error&) { continue; }
        }
        selected.push_back(f);
    }
    capture.frames = std::move(selected);
    if (capture.frames.size() < 4) throw std::runtime_error("至少需要 4 张带有效相机参数与位姿的参考照片。普通相册 ZIP 无法直接建库。");
    report(progress, .08, "已检查训练与测试划分，正在读取 GLB…");
    return capture;
}

cv::Mat decodeImage(const std::vector<unsigned char>& bytes, int reduce = 1) {
    if (bytes.size() > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("图片文件过大。");
    int w = 0, h = 0, n = 0;
    if (!stbi_info_from_memory(bytes.data(), int(bytes.size()), &w, &h, &n) || w < 1 || h < 1 || double(w) * h > 24e6) throw std::runtime_error("无法读取图片，或图片超过 2400 万像素。");
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> image(stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &n, 3), stbi_image_free);
    if (!image) throw std::runtime_error("参考照片或模型纹理损坏。");
    cv::Mat rgb(h, w, CV_8UC3, image.get()), result;
    cv::cvtColor(rgb, result, cv::COLOR_RGB2BGR);
    if (reduce > 1) cv::resize(result, result, cv::Size(std::max(1, w / reduce), std::max(1, h / reduce)), 0, 0, cv::INTER_AREA);
    return result;
}

void requireUpright(const std::vector<unsigned char>& data) {
    auto tiff = [&](size_t start, size_t length) {
        if (length < 8) return;
        auto* p = data.data() + start;
        bool little = p[0] == 'I' && p[1] == 'I', big = p[0] == 'M' && p[1] == 'M';
        if (!little && !big) return;
        auto u16 = [&](size_t i) -> unsigned { return little ? unsigned(p[i]) | unsigned(p[i + 1]) << 8 : unsigned(p[i]) << 8 | unsigned(p[i + 1]); };
        auto u32 = [&](size_t i) -> uint32_t { return little ? uint32_t(u16(i)) | uint32_t(u16(i + 2)) << 16 : uint32_t(u16(i)) << 16 | uint32_t(u16(i + 2)); };
        size_t directory = u32(4);
        if (directory > length - 2) throw std::runtime_error("参考照片 EXIF 信息损坏。");
        size_t count = u16(directory);
        if (count > (length - directory - 2) / 12) throw std::runtime_error("参考照片 EXIF 信息损坏。");
        for (size_t i = 0; i < count; ++i) {
            size_t entry = directory + 2 + 12 * i;
            if (u16(entry) == 274 && (u16(entry + 2) != 3 || u32(entry + 4) != 1 || u16(entry + 8) != 1))
                throw std::runtime_error("参考照片带有 EXIF 旋转。请使用已摆正的照片，并提供对应像素方向的相机参数与位姿。");
        }
    };
    if (data.size() >= 4 && data[0] == 255 && data[1] == 216) {
        size_t offset = 2;
        while (offset + 4 <= data.size() && data[offset] == 255) {
            while (offset < data.size() && data[offset] == 255) ++offset;
            if (offset >= data.size()) break;
            unsigned marker = data[offset++]; if (marker == 218 || marker == 217) break;
            if (offset + 2 > data.size()) break;
            size_t length = (size_t(data[offset]) << 8) | data[offset + 1];
            if (length < 2 || length > data.size() - offset) break;
            if (marker == 225 && length >= 8 && std::memcmp(data.data() + offset + 2, "Exif\0\0", 6) == 0) tiff(offset + 8, length - 8);
            offset += length;
        }
    } else if (data.size() >= 8 && std::memcmp(data.data(), "\x89PNG\r\n\x1a\n", 8) == 0) {
        size_t offset = 8;
        while (offset + 12 <= data.size()) {
            uint32_t length = 0; for (int i = 0; i < 4; ++i) length = (length << 8) | data[offset + i];
            if (length > data.size() - offset - 12) break;
            if (std::memcmp(data.data() + offset + 4, "eXIf", 4) == 0) tiff(offset + 8, length);
            offset += size_t(length) + 12;
        }
    }
}

Mesh readMesh(const fs::path& path, const BuildProgress& progress) {
    auto bytes = readBytes(path, 192 * 1024 * 1024);
    uint32_t jsonSize = 0;
    if (bytes.size() >= 20) for (int i = 0; i < 4; ++i) jsonSize |= uint32_t(bytes[12 + i]) << (8 * i);
    if (bytes.size() < 20 || jsonSize > 16 * 1024 * 1024) throw std::runtime_error("GLB 文件头无效，或 JSON 场景描述过大。");
    cgltf_options options = {}; cgltf_data* raw = nullptr;
    if (cgltf_parse(&options, bytes.data(), bytes.size(), &raw) != cgltf_result_success) throw std::runtime_error("GLB 无法解析，请选择完整的 glTF 2.0 二进制模型。");
    std::unique_ptr<cgltf_data, decltype(&cgltf_free)> data(raw, cgltf_free);
    if (raw->file_type != cgltf_file_type_glb || raw->buffers_count != 1 || raw->buffers[0].uri || raw->buffers[0].size > raw->bin_size || !raw->bin)
        throw std::runtime_error("模型必须是内嵌几何数据的 GLB，不能引用外部缓冲文件。");
    raw->buffers[0].data = const_cast<void*>(raw->bin);
    // cgltf does not own the memory buffer supplied to cgltf_parse.
    raw->buffers[0].data_free_method = cgltf_data_free_method_none;
    for (size_t i = 0; i < raw->buffer_views_count; ++i) {
        const auto& view = raw->buffer_views[i];
        if (view.buffer != &raw->buffers[0] || view.offset > raw->bin_size || view.size > raw->bin_size - view.offset)
            throw std::runtime_error("GLB 缓冲区索引或长度越界。");
    }
    for (size_t i = 0; i < raw->accessors_count; ++i) {
        const auto& accessor = raw->accessors[i];
        if (accessor.count < 1 || accessor.count > 6000000 || accessor.stride > 256 || accessor.offset > raw->bin_size)
            throw std::runtime_error("GLB 访问器数量、步长或偏移无效。");
    }
    if (raw->nodes_count > 10000 || raw->images_count > 128) throw std::runtime_error("GLB 场景节点或纹理过多。");
    for (size_t i = 0; i < raw->nodes_count; ++i) {
        std::set<const cgltf_node*> ancestors;
        for (auto node = &raw->nodes[i]; node; node = node->parent) {
            if (!ancestors.insert(node).second) throw std::runtime_error("GLB 节点包含循环引用。");
            if (ancestors.size() > 128) throw std::runtime_error("GLB 节点层级过深。");
        }
    }
    if (cgltf_validate(raw) != cgltf_result_success) throw std::runtime_error("GLB 的顶点、索引或场景结构无效。");
    std::set<const cgltf_node*> active;
    std::function<void(const cgltf_node*)> visit = [&](const cgltf_node* node) {
        if (!active.insert(node).second) throw std::runtime_error("GLB 场景重复引用同一个节点。");
        for (size_t i = 0; i < node->children_count; ++i) visit(node->children[i]);
    };
    if (raw->scene) for (size_t i = 0; i < raw->scene->nodes_count; ++i) visit(raw->scene->nodes[i]);
    else for (size_t i = 0; i < raw->nodes_count; ++i) if (!raw->nodes[i].parent) visit(&raw->nodes[i]);
    Mesh mesh; std::map<const cgltf_image*, cv::Mat> textures; size_t textureBytes = 0;
    for (const auto node : active) {
        if (!node->mesh) continue;
        if (node->skin || node->mesh->weights_count || node->has_mesh_gpu_instancing) throw std::runtime_error("请导入静态房间 GLB，不支持骨骼、变形或 GPU 实例化模型。");
        float transform[16]; cgltf_node_transform_world(node, transform);
        for (float v : transform) if (!std::isfinite(v)) throw std::runtime_error("GLB 节点变换无效。");
        for (size_t p = 0; p < node->mesh->primitives_count; ++p) {
            const auto& primitive = node->mesh->primitives[p];
            if (primitive.type != cgltf_primitive_type_triangles || primitive.has_draco_mesh_compression || primitive.targets_count) throw std::runtime_error("请导出未使用 Draco 压缩的静态三角面 GLB。");
            const cgltf_accessor *positions = nullptr, *colors = nullptr, *uv = nullptr;
            for (size_t a = 0; a < primitive.attributes_count; ++a) {
                const auto& attribute = primitive.attributes[a];
                if (attribute.type == cgltf_attribute_type_position) positions = attribute.data;
                if (attribute.type == cgltf_attribute_type_color && attribute.index == 0) colors = attribute.data;
                if (attribute.type == cgltf_attribute_type_texcoord && attribute.index == 0) uv = attribute.data;
            }
            if (!positions || positions->type != cgltf_type_vec3 || positions->count + mesh.vertices.size() > 2000000) throw std::runtime_error("GLB 缺少顶点，或超过 200 万顶点的手机限制。");
            for (const cgltf_accessor* accessor : std::array<const cgltf_accessor*, 4>{positions, colors, uv, primitive.indices})
                if (accessor && (accessor->is_sparse || (accessor->buffer_view && accessor->buffer_view->has_meshopt_compression))) throw std::runtime_error("请导出未使用稀疏或 Meshopt 压缩的 GLB。");
            if ((colors && colors->count != positions->count) || (uv && uv->count != positions->count)) throw std::runtime_error("GLB 顶点颜色或纹理坐标长度无效。");
            cv::Vec4f factor(1, 1, 1, 1); cv::Mat texture;
            const cgltf_texture_view* view = nullptr;
            if (primitive.material && primitive.material->has_pbr_metallic_roughness) {
                const auto& material = primitive.material->pbr_metallic_roughness;
                factor = cv::Vec4f(material.base_color_factor); view = &material.base_color_texture;
                if (view->texture && view->texcoord == 0 && view->texture->image && view->texture->image->buffer_view) {
                    const auto* image = view->texture->image->buffer_view;
                    auto begin = cgltf_buffer_view_data(image);
                    if (!begin || image->size > 64 * 1024 * 1024) throw std::runtime_error("GLB 纹理文件大小无效。");
                    auto id = view->texture->image; auto found = textures.find(id);
                    if (found != textures.end()) texture = found->second;
                    else {
                        texture = decodeImage(std::vector<unsigned char>(begin, begin + image->size), 2);
                        size_t size = texture.total() * texture.elemSize();
                        if (textureBytes + size > 64 * 1024 * 1024) { textures.clear(); textureBytes = 0; }
                        textures[id] = texture; textureBytes += size;
                    }
                }
            }
            int offset = int(mesh.vertices.size());
            for (size_t i = 0; i < positions->count; ++i) {
                if (i % 4096 == 0) checkpoint(progress);
                float v[4] = {}, c[4] = {1, 1, 1, 1}, t[2] = {};
                if (!cgltf_accessor_read_float(positions, i, v, 3)) throw std::runtime_error("GLB 顶点读取失败。");
                Vec point;
                for (int d = 0; d < 3; ++d) {
                    point[d] = transform[d] * v[0] + transform[4 + d] * v[1] + transform[8 + d] * v[2] + transform[12 + d];
                    if (!std::isfinite(point[d]) || std::abs(point[d]) >= 1e6) throw std::runtime_error("GLB 顶点坐标无效。");
                }
                if (colors && !cgltf_accessor_read_float(colors, i, c, 4)) throw std::runtime_error("GLB 顶点颜色读取失败。");
                cv::Vec4f rgba(c); for (int d = 0; d < 4; ++d) rgba[d] *= factor[d];
                if (uv && !texture.empty() && cgltf_accessor_read_float(uv, i, t, 2)) {
                    float x = t[0], y = t[1];
                    if (view->has_transform) {
                        auto& tr = view->transform;
                        float tx = tr.scale[0] * x, ty = tr.scale[1] * y;
                        x = std::cos(tr.rotation) * tx - std::sin(tr.rotation) * ty + tr.offset[0];
                        y = std::sin(tr.rotation) * tx + std::cos(tr.rotation) * ty + tr.offset[1];
                    }
                    if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("GLB 纹理坐标无效。");
                    auto pixel = texture.at<cv::Vec3b>(std::min(texture.rows - 1, int((y - std::floor(y)) * texture.rows)), std::min(texture.cols - 1, int((x - std::floor(x)) * texture.cols)));
                    for (int d = 0; d < 3; ++d) rgba[d] *= pixel[2 - d] / 255.f;
                }
                for (float& v : rgba.val) { if (!std::isfinite(v)) throw std::runtime_error("GLB 顶点颜色无效。"); v = std::clamp(v, 0.f, 1.f); }
                mesh.vertices.push_back(point); mesh.colors.push_back(rgba);
            }
            size_t indices = primitive.indices ? primitive.indices->count : positions->count;
            if (indices % 3 || indices / 3 + mesh.faces.size() > 2000000) throw std::runtime_error("GLB 三角面数量无效或超过手机限制。");
            for (size_t i = 0; i < indices; i += 3) {
                cv::Vec3i face;
                for (int j = 0; j < 3; ++j) {
                    size_t index = primitive.indices ? cgltf_accessor_read_index(primitive.indices, i + j) : i + j;
                    if (index >= positions->count) throw std::runtime_error("GLB 索引越界。");
                    face[j] = offset + int(index);
                }
                mesh.faces.push_back(face);
            }
        }
        report(progress, .14, "正在读取房间几何与颜色…");
    }
    if (mesh.faces.empty()) throw std::runtime_error("GLB 不包含可定位的三角面。");
    return mesh;
}

cv::Mat renderDepth(const Mesh& mesh, const Frame& frame, const BuildProgress& progress) {
    cv::Mat image(frame.size, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
    cv::Matx33d R; cv::Vec3d center;
    for (int i = 0; i < 3; ++i) { center[i] = frame.pose(i, 3); for (int j = 0; j < 3; ++j) R(i, j) = frame.pose(i, j); }
    std::vector<cv::Vec3d> vertices; vertices.reserve(mesh.vertices.size());
    for (const auto& v : mesh.vertices) vertices.push_back(R.t() * (cv::Vec3d(v) - center));
    size_t faceIndex = 0;
    for (const auto& face : mesh.faces) {
        if (++faceIndex % 512 == 0) checkpoint(progress);
        const auto &a = vertices[face[0]], &b = vertices[face[1]], &c = vertices[face[2]];
        if (std::min({a[2], b[2], c[2]}) <= .05) continue;
        double ax = a[0] / a[2] * frame.K(0, 0) + frame.K(0, 2), ay = a[1] / a[2] * frame.K(1, 1) + frame.K(1, 2);
        double bx = b[0] / b[2] * frame.K(0, 0) + frame.K(0, 2), by = b[1] / b[2] * frame.K(1, 1) + frame.K(1, 2);
        double cx = c[0] / c[2] * frame.K(0, 0) + frame.K(0, 2), cy = c[1] / c[2] * frame.K(1, 1) + frame.K(1, 2);
        double den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
        if (std::abs(den) < 1e-10) continue;
        // Clamp before converting to int; vertices very near the camera can project far outside the image.
        int xmin = int(std::clamp(std::ceil(std::min({ax, bx, cx})), 0., double(frame.size.width))), xmax = int(std::clamp(std::floor(std::max({ax, bx, cx})), -1., double(frame.size.width - 1)));
        int ymin = int(std::clamp(std::ceil(std::min({ay, by, cy})), 0., double(frame.size.height))), ymax = int(std::clamp(std::floor(std::max({ay, by, cy})), -1., double(frame.size.height - 1)));
        for (int y = ymin; y <= ymax; ++y) for (int x = xmin; x <= xmax; ++x) {
            double wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den;
            double wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den, wc = 1 - wa - wb;
            if (std::min({wa, wb, wc}) < -1e-7) continue;
            double z = 1 / (wa / a[2] + wb / b[2] + wc / c[2]);
            if (z > .05 && z < 10 && z < image.at<float>(y, x)) image.at<float>(y, x) = float(z);
        }
    }
    return image;
}

Reference extract(const Mesh& mesh, Frame frame, int features, const BuildProgress& progress) {
    auto bytes = readBytes(frame.image, 64 * 1024 * 1024); requireUpright(bytes);
    cv::Mat image = decodeImage(bytes);
    if (image.size() != frame.size) throw std::runtime_error("参考照片尺寸与相机参数不一致：" + frame.id);
    double scale = std::min(1., 960. / std::max(image.cols, image.rows));
    cv::Size size(int(std::round(image.cols * scale)), int(std::round(image.rows * scale)));
    if (size != image.size()) cv::resize(image, image, size, 0, 0, cv::INTER_AREA);
    double sx = double(size.width) / frame.size.width, sy = double(size.height) / frame.size.height;
    frame.K(0, 0) *= sx; frame.K(1, 1) *= sy;
    frame.K(0, 2) = (frame.K(0, 2) + .5) * sx - .5; frame.K(1, 2) = (frame.K(1, 2) + .5) * sy - .5; frame.size = size;
    Reference ref; ref.id = frame.id; ref.hash = picosha2::hash256_hex_string(bytes);
    std::vector<cv::KeyPoint> points; ref.descriptors = extractRootSIFT(image, features, points);
    if (ref.descriptors.rows > features) { ref.descriptors = ref.descriptors.rowRange(0, features).clone(); points.resize(features); }
    ref.world = cv::Mat::zeros(int(points.size()), 3, CV_32F); ref.valid = cv::Mat::zeros(int(points.size()), 1, CV_8U);
    if (points.empty()) return ref;
    cv::Mat depth = renderDepth(mesh, frame, progress);
    for (int i = 0; i < int(points.size()); ++i) {
        auto uv = points[i].pt; int x = int(std::nearbyint(uv.x)), y = int(std::nearbyint(uv.y));
        if (x < 1 || y < 1 || x >= size.width - 1 || y >= size.height - 1) continue;
        float z = depth.at<float>(y, x), lo = z, hi = z; bool valid = z > .15f && z < 5.f;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) { float v = depth.at<float>(y + dy, x + dx); valid &= std::isfinite(v); lo = std::min(lo, v); hi = std::max(hi, v); }
        if (!valid || hi - lo >= .15f) continue;
        auto p = frame.pose * cv::Vec4d((uv.x - frame.K(0, 2)) / frame.K(0, 0) * z, (uv.y - frame.K(1, 2)) / frame.K(1, 1) * z, z, 1);
        for (int d = 0; d < 3; ++d) ref.world.at<float>(i, d) = float(p[d]);
        ref.valid.at<unsigned char>(i) = 1;
    }
    return ref;
}

void writeDisplayMesh(const Mesh& full, const fs::path& path, const BuildProgress& progress) {
    Mesh display;
    for (float voxel = .01f; voxel <= 5.f; voxel *= 1.5f) {
        std::map<std::array<int64_t, 3>, int> cells; std::vector<int> inverse, counts; display = {};
        for (size_t i = 0; i < full.vertices.size(); ++i) {
            if (i % 8192 == 0) checkpoint(progress);
            const auto& v = full.vertices[i]; std::array<int64_t, 3> key = {int64_t(std::floor(v[0] / voxel)), int64_t(std::floor(v[1] / voxel)), int64_t(std::floor(v[2] / voxel))};
            auto entry = cells.emplace(key, int(cells.size())); int id = entry.first->second;
            if (entry.second) { display.vertices.push_back(Vec(0, 0, 0)); display.colors.push_back(cv::Vec4f(0, 0, 0, 0)); counts.push_back(0); }
            display.vertices[id] += v; display.colors[id] += full.colors[i]; ++counts[id]; inverse.push_back(id);
        }
        for (size_t i = 0; i < counts.size(); ++i) { display.vertices[i] /= float(counts[i]); display.colors[i] /= float(counts[i]); }
        std::set<std::array<int, 3>> seen;
        size_t faceIndex = 0;
        for (const auto& f : full.faces) {
            if (++faceIndex % 8192 == 0) checkpoint(progress);
            cv::Vec3i face(inverse[f[0]], inverse[f[1]], inverse[f[2]]);
            if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2]) continue;
            std::array<int, 3> key = {face[0], face[1], face[2]}; std::sort(key.begin(), key.end());
            if (seen.insert(key).second) display.faces.push_back(face);
        }
        report(progress, .92, "正在保存手机显示模型…");
        if (display.vertices.size() <= 200000 && display.faces.size() <= 300000) break;
    }
    if (display.vertices.empty() || display.faces.empty() || display.vertices.size() > 200000 || display.faces.size() > 300000) throw std::runtime_error("模型无法缩减到手机显示限制。");
    std::vector<Vec> normals(display.vertices.size(), Vec(0, 0, 0));
    for (const auto& f : display.faces) { auto n = (display.vertices[f[1]] - display.vertices[f[0]]).cross(display.vertices[f[2]] - display.vertices[f[0]]); for (int j : f.val) normals[j] += n; }
    std::ofstream output(path, std::ios::binary); output.write("RMSH0001", 8);
    auto u32 = [&](uint32_t value) { for (int i = 0; i < 4; ++i) output.put(char((value >> (8 * i)) & 255)); };
    auto f32 = [&](float v) { uint32_t bits; std::memcpy(&bits, &v, 4); u32(bits); };
    u32(uint32_t(display.vertices.size())); u32(uint32_t(display.faces.size()));
    for (size_t i = 0; i < display.vertices.size(); ++i) {
        double length = cv::norm(normals[i]); if (length > 0) normals[i] /= float(length);
        for (float v : display.vertices[i].val) f32(v);
        for (float v : normals[i].val) f32(v);
        for (float v : display.colors[i].val) f32(v);
    }
    for (const auto& f : display.faces) for (int i : f.val) u32(uint32_t(i));
    output.close(); if (!output) throw std::runtime_error("显示模型保存失败，请检查手机剩余空间。");
}
}

void buildRoom(const std::string& glbPath, const std::string& referenceRoot, const std::string& outputDirectory,
               const BuildOptions& options, const BuildProgress& progress) {
    if (options.maxReferences < 4 || options.maxReferences > 200 || options.featuresPerImage < 64 || options.featuresPerImage > 2000 || options.name.empty() || options.name.size() > 240)
        throw std::runtime_error("房间名称或参考库设置超出手机限制。");
    fs::path output(outputDirectory); fs::create_directories(output);
    for (const auto& file : {"map.yml", "mesh.bin", "map.yml.part", "mesh.bin.part"}) if (fs::exists(output / file)) throw std::runtime_error("请使用新的房间目录，不能覆盖已有参考库。");
    try {
        report(progress, .02, "正在识别参考数据格式…");
        auto capture = readCapture(referenceRoot, options.maxReferences, progress);
        auto mesh = readMesh(glbPath, progress); std::vector<Reference> references;
        cv::Mat samples;
        for (size_t i = 0; i < capture.frames.size(); ++i) {
            report(progress, .16 + .57 * i / capture.frames.size(), "正在处理参考照片 " + std::to_string(i + 1) + "/" + std::to_string(capture.frames.size()));
            auto ref = extract(mesh, capture.frames[i], options.featuresPerImage, progress);
            if (ref.descriptors.rows < 12 || cv::countNonZero(ref.valid) < 12) continue;
            // Bound vocabulary training memory independent of the number of reference features.
            for (int j = 0; j < ref.descriptors.rows; j += std::max(1, ref.descriptors.rows / 160)) if (samples.rows < 32000) samples.push_back(ref.descriptors.row(j));
            references.push_back(std::move(ref));
        }
        if (references.size() < 4) throw std::runtime_error("不足 4 张参考照片能对应到 GLB。请检查模型与参考位姿的坐标、单位及纹理覆盖。");
        report(progress, .75, "正在建立本机图像检索词典…");
        int words = std::min(128, samples.rows); cv::Mat labels, centers; cv::setRNGSeed(42);
        cv::kmeans(samples, words, labels, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .001), 1, cv::KMEANS_PP_CENTERS, centers);
        cv::Mat counts = cv::Mat::zeros(int(references.size()), words, CV_32F), idf = cv::Mat::ones(1, words, CV_32F);
        for (int i = 0; i < int(references.size()); ++i) {
            report(progress, .82 + .06 * i / references.size(), "正在生成参考图检索索引…");
            std::vector<cv::DMatch> assigned; cv::BFMatcher(cv::NORM_L2).match(references[i].descriptors, centers, assigned);
            for (const auto& match : assigned) counts.at<float>(i, match.trainIdx) += 1.f;
        }
        for (int j = 0; j < words; ++j) { int found = 0; for (int i = 0; i < counts.rows; ++i) found += counts.at<float>(i, j) > 0; idf.at<float>(j) = float(std::log((counts.rows + 1.) / (found + 1.)) + 1.); }
        cv::Mat vectors = cv::Mat::zeros(counts.size(), CV_32F);
        for (int i = 0; i < counts.rows; ++i) {
            for (int j = 0; j < words; ++j) vectors.at<float>(i, j) = std::sqrt(counts.at<float>(i, j) / std::max(1, references[i].descriptors.rows)) * idf.at<float>(j);
            double norm = cv::norm(vectors.row(i)); if (norm > 0) vectors.row(i) /= norm;
        }
        report(progress, .89, "正在保存房间参考库…");
        cv::FileStorage map((output / "map.yml.part").string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML | cv::FileStorage::BASE64);
        if (!map.isOpened()) throw std::runtime_error("无法写入房间参考库。");
        // FileStorage's << operator treats strings beginning with { or [ as
        // structural tokens. Camera JSON and user names must be literal values.
        map.write("format", "roomloc-map-v1"); map.write("name", options.name);
        map.write("units", "meters"); map.write("camera_convention", "opencv-camera-to-world");
        map.write("dataset_format", capture.format); map.write("dataset_camera_json", capture.camera);
        map.write("nfeatures", options.featuresPerImage);
        map << "centers" << centers << "idf" << idf << "vectors" << vectors << "references" << "[";
        for (const auto& ref : references) map << "{" << "id" << ref.id << "image_sha256" << ref.hash << "descriptors" << ref.descriptors << "world_points" << ref.world << "valid" << ref.valid << "}";
        map << "]"; map.release();
        if (fs::file_size(output / "map.yml.part") > 256 * 1024 * 1024) throw std::runtime_error("参考库超过手机限制，请减少参考照片数量。");
        writeDisplayMesh(mesh, output / "mesh.bin.part", progress);
        report(progress, .98, "正在验证新房间…");
        references.clear(); references.shrink_to_fit(); mesh = {}; samples.release();
        Localizer validation((output / "map.yml.part").string());
        report(progress, .99, "参考库已生成，正在保存房间…");
        fs::rename(output / "map.yml.part", output / "map.yml"); fs::rename(output / "mesh.bin.part", output / "mesh.bin");
    } catch (...) {
        for (const auto& file : {"map.yml", "mesh.bin", "map.yml.part", "mesh.bin.part"}) { std::error_code ec; fs::remove(output / file, ec); }
        throw;
    }
}
}
