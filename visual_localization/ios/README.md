# iPhone 离线照片定位

这是可在 Xcode 中打开的原生 iOS 工程。房间准备在电脑完成，照片定位在手机的 C++ / OpenCV 核心中运行，不调用电脑、云端或 Python 服务。

当前状态：已完成源码和工程配置；C++ 核心在 Linux 上编译并通过几何及跨语言测试；Swift 语法、工程引用、Info.plist 和共享 Scheme 已检查。**当前开发环境没有 Xcode，因此尚未完成 Apple SDK 编译、签名或 iPhone 真机验证。** 首次使用请先运行下文的 Mac 构建检查，并记录出现的编译或设备问题。

## 1. 安装到自己的 iPhone

需要 Mac、Xcode 15 或更新版本、iOS 16 或更新版本的 iPhone，以及自己的 Apple 账号。Xcode 版本还必须支持手机当前的 iOS 版本。

1. 下载包含此目录的整个仓库，保留 `RoomLocate/`、`Native/` 和 `RoomLocate.xcodeproj` 的相对位置。
2. 用 Xcode 打开 `RoomLocate.xcodeproj`。
3. 首次打开时等待 Swift Package Manager 解析和下载 OpenCV。工程固定使用 `yeatse/opencv-spm` 的 **4.12.0**，其包清单校验预编译 XCFramework 的 SHA-256。
4. 在 Xcode 的 Settings → Apple Accounts 登录自己的账号。
5. 选择 RoomLocate target → Signing & Capabilities，启用 Automatically manage signing，Team 选择自己的 Personal Team 或开发者团队。
6. 把默认 Bundle Identifier `com.example.m202a.RoomLocate` 改为自己的唯一标识，例如 `com.yourname.roomloc`。不要在仓库中提交账号凭据或签名证书。
7. 连接并信任 iPhone，按 Xcode 提示完成配对，在手机的“设置 → 隐私与安全性”开启开发者模式。
8. 顶部运行目标选择真实 iPhone，点击 Run。

Xcode 安装依赖和首次签名需要联网。完成安装和房间导入后，照片定位可断开电脑并离线使用。免费 Personal Team 的开发描述文件通常需要每 7 天重新通过 Xcode 构建安装；付费或学校开发团队按其签名规则处理。

不需要先上架 App Store。这个仓库不包含为某个用户签名的 IPA；签名必须使用安装者自己的 Apple 账号在 Xcode 中完成。

## 2. 从电脑导出房间

进入仓库的 `visual_localization/`，运行：

```bash
bash run_web.command
```

在网页添加 GLB 与参考数据 ZIP，等待房间准备好。点击 **导出 iPhone 房间包**，完成后点击下载，得到 `room.roomloc`。

更新已有电脑程序时请保留原有的 `data/` 和 `outputs/`，已有房间与参考库可以继续使用。若下载到全新的文件夹，则需要重新导入模型与参考数据，或使用下方命令指向已有数据和数据库。

已经有参考数据库时，也可以通过命令行导出：

```bash
python -m wifi_localization export-mobile \
  --data /path/to/prepared-room \
  --db /path/to/database \
  --name "Room name" \
  --output outputs/room.roomloc
```

网页导出的房间数据库通常位于 `outputs/web/rooms/<room-id>/output/database`，对应的 `--data` 为同一房间目录下的 `data`。已有的 Polycam 或 7-Scenes 数据库也可导出。

导出包只包含参考帧的特征、参考特征对应的三维点、检索词典和简化的彩色显示模型。不会打包原始参考照片、查询照片、查询位姿、测试深度或增强模型权重。官方训练/测试划分继续保留；低纹理、无法提取至少两个特征的参考帧会被跳过，最终至少需要四张可用参考图。

## 3. 在手机定位

1. 通过 AirDrop、iCloud Drive 或其他文件传输方式，把 `.roomloc` 文件保存到手机。
2. 在 Room Locate 中点击“导入”，选择文件；也可从“文件”中打开房间包。
3. 等待房间导入，三维模型出现后选择“拍照”或“相册”。
4. 默认使用自动焦距估算。若有针对同一镜头、模式、分辨率和方向的标定 JSON，可以导入并选择“已导入标定”。
5. 点击“定位这张照片”。成功返回后显示红色相机位置与镜头朝向、X/Y/Z、内点数和处理时间。
6. “导出位姿 JSON”可保存或分享本次结果。照片仅在当前会话中处理，不自动发送到外部服务。

7-Scenes 参数只对原始、未裁剪的 640×480 数据集照片开放，不能直接用于普通 iPhone 照片。更换房间时会重置相机模式，并加载该房间单独保存的标定。更换照片、房间或相机模式后，旧的定位标记和分享链接会清除。

## 4. 本版本边界

- 手机端为 RootSIFT、TF-IDF 检索、互相匹配、2D-3D 对应和 PnP/RANSAC。通过内点数、比例、图像分布、正深度和三维分布检查后才返回位姿。
- 自动焦距使用多个候选值，仍属于估算。手机第一版尚未移植桌面端的联合焦距优化及 ALIKED/LightGlue 回退，不能把桌面增强策略的成功率当成手机成功率。
- 当前是单张照片定位。ARKit 连续跟踪、Wi-Fi 采集以及实时视频定位尚未接入。
- 手机显示模型是从同一个 GLB 导出的简化顶点颜色网格。其坐标保持不变，视觉纹理细节可能比原 GLB 少。手机导出的是位姿 JSON，桌面端仍保留带标记 GLB 导出。
- 包内参考 YAML 上限 256 MiB，显示模型上限 64 MiB；最多 1000 张参考帧、总计 400000 个特征。大型场景应在电脑端减少参考帧或特征后重新导出。实际速度、内存和精度需要在目标 iPhone 上测量。
- `.roomloc` 包仍包含房间几何和特征，应像自己的房间模型一样保存。仓库忽略房间包和运行输出。

## 5. 检查和测试

在 Mac 的这个 `ios/` 目录运行：

```bash
bash check_on_mac.command
```

脚本会解析固定的 OpenCV 依赖，执行不需要开发者签名的模拟器构建。模拟器构建通过后，再用 Xcode 对真实 iPhone 签名安装。相机、处理耗时和手机内存表现要以真机为准。

C++ 核心可在有 OpenCV 4 开发库的 Mac/Linux 上独立测试：

```bash
cmake -S . -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

从 `visual_localization/` 运行 Python 与原生导出集成测试：

```bash
ROOMLOC_NATIVE_CLI="$PWD/ios/build/native/roomloc_native" \
  python -m unittest discover -s tests -p test_mobile.py -v
```

测试包括独立查询视角的已知位姿恢复、Python 与 C++ 结果一致性、空白照片失败、重复匹配只计一票、训练/测试隔离、错误特征不覆盖旧文件，以及本机 HTTP 导出与下载。可移植测试通过不等于 iPhone 实际定位精度已验证。

工程已经生成并提交。只有新增源文件或调整工程配置时，才需要运行 `python3 tools/generate_project.py` 重新生成；重新生成会清除在工程中手动设置的 Team 和 Bundle Identifier，操作前请记录自己的设置。

依赖来源：[OpenCV](https://github.com/opencv/opencv)、[OpenCV-SPM 4.12.0](https://github.com/yeatse/opencv-spm/tree/4.12.0)。SPM 包提供者是第三方维护者，使用 OpenCV 预编译二进制；许可证与依赖清单随包保留。
