from config import Config
from clarifier_service import ClarifierService

def main():
    # 验证配置
    try:
        Config.validate()
        print("环境准备完成，DeepSeek API Key 已加载。")
    except ValueError as e:
        print(e)
        return

    # 初始化服务，使用默认最大追问轮数
    clarifier = ClarifierService()
    print("分类器链创建成功！")
    print("问题生成器链创建成功！")
    print("-" * 50)

    while True:
        print("\n请输入您的问题（输入 'quit' 或 'exit' 退出）:")
        user_input = input("> ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break

        if not user_input:
            print("请输入有效的问题。")
            continue

        try:
            clarifier.run_clarifier_flow(user_input)
        except Exception as e:
            print(f"处理问题时出错: {e}")

        print("\n" + "="*50)

if __name__ == "__main__":
    main()