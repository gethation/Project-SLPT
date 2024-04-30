import os
import multiprocessing

# 要执行的.py文件的路径
script_path = "/content/Datafactory/implement.py"

# 定义一个函数，用于执行脚本
def run_script(script_path):
    exit_code = os.system("python " + script_path)
    return exit_code

if __name__ == "__main__":
    # 创建一个进程池，同时运行5个进程
    pool = multiprocessing.Pool(processes = 12)
    
    # 使用进程池同时执行脚本
    results = pool.map(run_script, [script_path] * 12)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 处理结果
    for i, exit_code in enumerate(results):
        if exit_code == 0:
            print(f"第{i+1}次执行成功！")
        else:
            print(f"第{i+1}次执行失败。")