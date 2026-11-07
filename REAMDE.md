# 定位结果对接

定位成功后，可点击桌面界面的 **Pose JSON** 下载结果。`RoomLocate_pose_example.json` 是从 `desktop.11.7.1` 的一次实际定位输出中截取的核心字段。应用可以按下表读取，完整结果还包含匹配和相机参数诊断。

| 字段 | 含义 |
|---|---|
| `success` | 为 `true` 时读取 `pose` |
| `pose.position_m` | 相机在房间模型坐标系中的 `[x, y, z]`，单位米 |
| `pose.quaternion_xyzw` | 相机到房间的旋转，四元数顺序为 `[x, y, z, w]` |
| `pose.forward_world` | 相机朝向在房间坐标系中的单位向量，可用于绘制箭头 |
| `pose.T_world_camera_cv` | 4×4 相机到房间的矩阵，JSON 外层是行。按 `p_room = T_world_camera_cv × p_camera` 计算，点使用齐次列向量 `[x, y, z, 1]` |

手机端显示同一 GLB 时，可把标记放在模型根节点下，使用 `position_m` 作为它的本地位置。完整相机姿态使用 OpenCV 相机轴，局部 +X 向右、+Y 向下、+Z 向前。只显示位置和朝向时，读取 `position_m` 和 `forward_world` 即可。

桌面接口通过 `POST /api/localize` 接收 `multipart/form-data`，照片字段名为 `image`。提交成功返回 HTTP 202 和任务 `id`。查询 `GET /api/jobs/{id}`，等 `status` 为 `done` 后检查 `result.success`，从 `result.pose.position_m` 取位置。下载的 Pose JSON 直接使用 `result` 这一层。定位失败时 `success` 为 `false`，界面可根据 `reason` 显示失败状态。

当前桌面服务监听 `127.0.0.1`。手机可先读取导出的样例验证坐标显示，真机联调接口时需要手机可访问的服务入口。
