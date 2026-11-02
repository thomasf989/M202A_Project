# iPhone 本机建库与照片定位

Room Locate 可直接导入已有 **GLB + 参考数据 ZIP**，在 iPhone 上解包、建立定位参考库、处理查询照片和显示位姿。不需要扫描房间、电脑 Python 服务或云端计算。原来的 `.roomloc` 房间包仍可导入。

开发状态：可移植 C++ 建库与定位核心已在 Linux 编译并测试；包含 Swift ZIP 测试和 Mac 构建检查脚本。**本开发环境没有 Xcode、Swift 工具链或 Apple SDK，尚未完成本版本的 Swift 测试、Apple SDK 编译、签名与 iPhone 真机验证。源码更新不等于已经发布可直接安装的 App。**

## 1. 全程在手机上使用

1. 将对应房间的 `.glb` 模型和参考 `.zip` 保存到 iPhone 的“文件”App。可以直接在手机浏览器下载，或放在 iCloud Drive。云端文件需要下载完成才能离线使用。
2. 打开 Room Locate，点击 **新建房间**，或 **导入 → GLB 和参考数据**。
3. 输入房间名称，分别选择 **GLB 模型** 和 **参考数据 ZIP**。
4. 选择最多使用的参考照片数量，点击 **在手机上准备房间**。默认最多 120 张，可调至 1000 张；超出时均匀抽取训练/参考帧。更多照片会增加耗时，较大的参考库会自动减少每张图的特征预算。
5. 保持 App 在前台，等待进度完成。可以取消；未完成的房间会清理，已有房间不受影响。失败后可在当前会话保留所选文件并重试。
6. 模型出现后，点击 **拍照**、**相册** 或 **文件** 选择查询照片，再点击 **定位这张照片**。**文件** 会打开系统文件选择器，可进入“在我的 iPhone 上”或已配置的文件提供方，选择 JPG、PNG、HEIC 等图片。7-Scenes 的 `frame-*.color.png` 可直接从文件夹选择，无需先存入相册。云端照片需先下载；每次选择一张照片，文件上限 128 MiB、6500 万像素。
7. 成功后显示红色相机的位置与朝向，以及 X/Y/Z、内点数和耗时。点击 **导出位姿 JSON** 保存或分享结果。

准备完成的房间保存在 App 本地，正常退出、重启手机后仍可选择。删除 App 会移除其本地数据。App 不会自动上传照片或模型。建库和定位均不需要电脑保持开机，也不需要连接本地服务器。

## 2. 支持的参考数据

GLB 与参考相机位姿必须对应同一房间、同一坐标系，坐标单位为米。至少需要 4 张有纹理且能对应到模型几何的有效参考照片。**普通相册照片 ZIP 不包含相机参数和位姿，不能直接用来建库。**

| 格式 | ZIP 内需要的内容 | 测试数据处理 |
| --- | --- | --- |
| 原始 7-Scenes，无划分文件 | `frame-*.color.png` 和对应的 `frame-*.pose.txt`；可直接选择单个 `seq-XX.zip`，或将多个序列放入一个 ZIP | 没有 `TrainSplit.txt` 时，所有上传的 RGB 帧均作为训练参考候选；单独的 `TestSplit.txt` 不过滤照片 |
| 原始 7-Scenes，有明确划分 | `TrainSplit.txt` 和所列序列的 RGB、位姿；`TestSplit.txt` 可选；支持外层 ZIP 中嵌套一层 `seq-XX.zip` | 只展开/读取训练序列；提供测试划分时检查训练/测试不重叠；测试序列可以不提供 |
| 已准备的 7-Scenes | `seven_scenes.json`、训练 RGB 照片和位姿 | 验证训练/测试划分，只使用 `train`；测试照片和深度可以不提供 |
| Polycam 原始数据 | `optimized_roomplan.json`、`keyframes/corrected_cameras/*.json`、`keyframes/corrected_images/*.jpg` | 使用校正照片与相机；默认正向 alignment transform |
| 通用照片与位姿 | `references.json`，以及其中指定的 JPG/PNG 参考照片 | 只使用 `reference` / `train`，跳过 `test` |

一个 ZIP 可以有一层或多层外部文件夹，但只能包含一个房间的数据。通用格式的结构见 [references.json 示例](../wifi_localization/web/reference-format.json)。参考照片应已去畸变、以正确方向存储；相机参数和位姿必须针对这些实际像素，带 EXIF 旋转的参考图片会被拒绝。

不提供 `TrainSplit.txt` 时，App 不按官方序列划分猜测训练和测试集。请只上传想用于建库的序列；需要独立评估的查询照片另外保留。每张原始 RGB 照片仍须有同名 `.pose.txt`，不需要 `.depth.png`。多个序列中的同名帧会使用不同编号。已有清单中的 `test` 标记和明确提供的 `TrainSplit.txt` 仍然生效。

手机建库从 GLB 在 RGB 相机像素网格中渲染深度，再建立图像与三维点的对应。它不使用未配准的 Kinect 原始深度；Polycam 的手机建库也使用模型渲染深度，与桌面 Polycam 传感器深度路径不同，需要独立检查精度。原始 7-Scenes 默认使用近似 RGB 焦距 525 和 `diag(1,-1,-1,1)` 的世界到 GLB 变换；自定义对齐或内参请使用准备好的 `seven_scenes.json` 或通用位姿清单。

7-Scenes 查询参数只适用于原始、未裁剪的 640×480 数据集照片。普通 iPhone 照片使用自动焦距或对应镜头的标定 JSON。更换房间、照片或相机模式后会清除旧定位结果。

## 3. 安装与发布

手机端使用流程已经不依赖电脑准备房间。**App 的开发、编译和签名仍需要 Mac/Xcode 或配置好的云端构建服务。** 本仓库没有已签名 IPA，也没有已发布的 TestFlight 安装链接。

开发安装需要 Xcode 15 或更新版本、iOS 16 或更新版本的 iPhone，以及自己的 Apple 账户；Xcode 还必须支持手机的 iOS 版本。

1. 下载完整仓库，保留 `RoomLocate/`、`Native/`（包括 `vendor/`）与 `RoomLocate.xcodeproj` 的相对位置。
2. 打开 `RoomLocate.xcodeproj`，等待固定版本的 OpenCV-SPM **4.12.0** 和 ZIPFoundation **0.9.20** 解析完成。
3. 在 Xcode → Settings → Apple Accounts 登录自己的账户。
4. 选择 RoomLocate target → Signing & Capabilities，开启 Automatically manage signing，Team 选择自己的团队。
5. 将默认 Bundle Identifier `com.example.m202a.RoomLocate` 换成唯一标识，例如 `com.yourname.roomloc`。不要提交账号凭据或签名证书。
6. 连接并信任 iPhone，开启手机开发者模式；运行目标选择真机，点击 Run。

免费 Personal Team 描述文件 7 天到期，需要重新构建安装。如果要求安装和更新时也只操作手机，应由开发团队通过 TestFlight 或 App Store 发布。TestFlight 每个构建最长可测试 90 天，由开发端提供新构建后可在手机上更新；它不是永久签名方案。[Apple TestFlight 说明](https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/)。

### 构建时找不到 ZIPFoundation

若出现 `Unable to resolve module dependency: 'ZIPFoundation'`，先检查编译配置和包解析状态。仓库已声明并链接 ZIPFoundation 0.9.20；Swift 包需要在首次构建时下载完成。

1. 使用更新后的完整工程。Debug 的 **Build Active Architecture Only** 应为 **Yes**，Release 为 **No**。旧版工程漏设了 Debug 的此项配置，可能造成 App 与 Swift 包在 Apple silicon 模拟器上构建不同架构。可直接在 RoomLocate target → Build Settings → 搜索 `ONLY_ACTIVE_ARCH`，展开后将 Debug 改为 Yes。
2. 选择 **File → Packages → Resolve Package Versions**，等待包解析完成。
3. 确认工程的 **Package Dependencies** 中有 ZIPFoundation 0.9.20，且 RoomLocate target 的 **General → Frameworks, Libraries, and Embedded Content** 中关联了 ZIPFoundation。若依赖缺失，使用 **File → Add Package Dependencies**，输入 `https://github.com/weichsel/ZIPFoundation.git`，选择精确版本 0.9.20，并将产品加入 RoomLocate target。
4. 选择 **Product → Clean Build Folder**（⌘⇧K），再运行（⌘R）。

如果仍失败，请展开完整的 `Ignoring file` / `Module file` 警告，并查看构建日志中 ZIPFoundation 的第一条错误，以区分架构、包下载和缓存问题。不要仅凭这条模块错误断定包没有下载。本次配置修复尚未在此 Linux 环境完成 Apple SDK 构建。

依据：[Apple silicon 架构设置](https://developer.apple.com/documentation/technotes/tn3117-resolving-build-errors-for-apple-silicon)、[添加并关联 Swift 包](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app)。

## 4. 兼容已有房间包

点击 **导入 → 已有 .roomloc 房间包**，选择过去导出的文件即可。也可以从“文件”App 打开房间包。

电脑导出仍然可用，但只是可选方式：在 `visual_localization/` 运行 `bash run_web.command`，添加 GLB 和参考 ZIP，完成后点击 **导出 iPhone 房间包**。

已有参考数据库也可导出：

```bash
python -m wifi_localization export-mobile \
  --data /path/to/prepared-room \
  --db /path/to/database \
  --name "Room name" \
  --output outputs/room.roomloc
```

包内包含参考特征、三维对应点、检索词典与简化显示网格，不包含查询照片、查询位姿或增强匹配权重。

## 5. 本版本边界与大小限制

- 建库和定位均使用本机 C++ / OpenCV。定位为 RootSIFT、TF-IDF 检索、互相匹配、2D-3D 对应及 PnP/RANSAC，通过几何支持与分布检查后才返回位姿。
- 尚未移植桌面 ALIKED/LightGlue 回退和联合焦距优化；不能把桌面增强匹配的成功率视为手机成功率。自动焦距仍为估算。
- 当前为单张照片定位，不包含扫描、ARKit 连续跟踪、视频定位或 Wi-Fi 采集。
- GLB 必须是静态三角面模型，几何数据内嵌，不支持 Draco、Meshopt、稀疏访问器、骨骼、变形或 GPU 实例化。显示模型保留原坐标并简化为顶点颜色网格，纹理细节可能减少。
- GLB 上限 192 MiB、200 万顶点、200 万三角面，JSON 场景描述上限 16 MiB；参考图片上限 2400 万像素。
- 输入 ZIP 上限 4 GiB，累计解包上限 6 GiB、80000 个条目；使用流式解包、路径与重复项检查、CRC 校验和符号链接拒绝。准备房间需要足够临时空间。
- 手机建库默认最多选 120 张，可调至 1000 张。默认每张最多 1500 个特征，参考总预算为 300000 个特征；1000 张时每张最多 300 个。查询照片仍使用原来的 1500 特征预算。保存前先验证新库，再提交房间记录。已有 `.roomloc` 兼容最多 1000 张参考帧、400000 个特征的限制。
- 保存的参考 YAML 上限 256 MiB，显示网格上限 64 MiB、200000 顶点和 300000 三角面。速度、峰值内存、热量和定位精度需要在目标 iPhone 实测，不承诺固定分钟数。

## 6. 构建检查与测试

Mac 上在这个 `ios/` 目录运行：

```bash
bash check_on_mac.command
```

脚本运行 Swift ZIP 测试、模拟器构建和不签名的真机 Release 构建。通过后，再配置自己的签名进行真机安装或发布。相机权限、方向、文件选择器、前后台切换、实际手机内存、耗时和精度仍需真机检查。

也可单独运行 ZIP 测试：

```bash
swift test
```

测试有效 ZIP、路径遍历、符号链接、重复路径、CRC 损坏、取消清理、已有目录保护、大小限制、有划分时的训练/测试隔离，以及无划分时直接导入序列或展开全部嵌套序列 ZIP。

可移植 C++ 核心可在安装 OpenCV 4 开发库的 Mac/Linux 上编译：

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

从 `visual_localization/` 运行全部 Python / C++ 集成测试：

```bash
ROOMLOC_NATIVE_CLI="$PWD/ios/build/native/roomloc_native" \
  python -m unittest discover -s tests -v
```

`test_phone_build.py` 直接调用与 App 相同的本机建库核心，检查独立查询视角恢复、参考照片哈希、OpenGL/OpenCV 坐标转换、GLB 节点变换、7-Scenes 有/无划分、多序列重名帧、缺失位姿、1000 张参考库的建库及重新加载、Polycam 校正相机、均匀抽帧、无效输入与取消回滚。1000 张压力用例复用四张合成纹理照片，验证数量、特征和保存边界，不代表 1000 个真实视角的定位精度。`test_mobile.py` 保留桌面导出到手机核心的兼容检查。可移植测试通过不等于 Apple SDK 构建或 iPhone 精度已经通过。

新增源码或修改依赖后运行 `python3 tools/generate_project.py` 更新工程。重新生成会清除手动配置的 Team 与 Bundle Identifier，操作前记录自己的设置。建库 C++ 文件在 Debug 中也开启优化，避免几何渲染循环因无优化而明显变慢。

依赖：[OpenCV](https://github.com/opencv/opencv)、[OpenCV-SPM](https://github.com/yeatse/opencv-spm/tree/4.12.0)、[ZIPFoundation](https://github.com/weichsel/ZIPFoundation/tree/0.9.20)，以及有固定版本和校验值的 [C++ 依赖清单](Native/vendor/README.md)。App 的“开源许可”包含相应声明。
