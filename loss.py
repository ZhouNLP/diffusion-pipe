import os
import subprocess
import time

# 配置
TENSORBOARD_PORT = 6006
TENSORBOARD_HOST = "localhost"

def find_event_file(directory):
    """在指定目录下查找第一个 TensorBoard 事件文件"""
    if not os.path.isdir(directory):
        print(f"错误：目录 {directory} 不存在，请检查路径！")
        return None

    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("events.out.tfevents"):
                return os.path.join(root, file)
    print(f"错误：在目录 {directory} 中未找到 TensorBoard 事件文件（events.out.tfevents*）！")
    return None

def start_tensorboard(event_file):
    """启动 TensorBoard 并打印访问地址"""
    if not event_file or not os.path.exists(event_file):
        print(f"错误：文件 {event_file} 不存在，请确保路径正确！")
        return False

    # 获取事件文件所在的目录
    log_dir = os.path.dirname(event_file)
    print(f"启动 TensorBoard，日志目录: {log_dir}")

    # 启动 TensorBoard
    try:
        tensorboard_cmd = [
            "tensorboard",
            f"--logdir={log_dir}",
            f"--port={TENSORBOARD_PORT}",
            f"--host={TENSORBOARD_HOST}"
        ]
        process = subprocess.Popen(
            tensorboard_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 等待 TensorBoard 启动
        time.sleep(2)

        # 检查 TensorBoard 是否启动成功
        if process.poll() is not None:
            error = process.stderr.read()
            print(f"TensorBoard 启动失败: {error}")
            return False

        # 打印访问地址
        tensorboard_url = f"http://{TENSORBOARD_HOST}:{TENSORBOARD_PORT}"
        print(f"TensorBoard 已启动，请通过以下地址访问：")
        print(f"  {tensorboard_url}")
        print("按 Ctrl+C 退出程序。")

        # 保持程序运行，直到用户手动终止
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在关闭 TensorBoard...")
            process.terminate()
            process.wait()
            print("程序已退出。")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"TensorBoard 启动失败: {e}")
        return False

def main():
    # 提示用户输入目录
    print("请粘贴包含 TensorBoard 事件文件的目录路径（例如 /root/diffusion-pipe/wan2.1/output），然后按回车：")
    directory = input().strip()

    # 处理路径（去除可能的多余引号或空格）
    directory = directory.strip("'").strip('"').strip()

    # 查找事件文件
    event_file = find_event_file(directory)
    if event_file:
        # 启动 TensorBoard
        start_tensorboard(event_file)

if __name__ == "__main__":
    main()