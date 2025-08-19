from flask import Flask, request, jsonify
from clarifier_service import ClarifierService
from config import Config
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


class APIClarifierService:
    def __init__(self):
        self.clarifier = ClarifierService()
        self.active_sessions = {}  # 存储活跃的对话会话

    def start_clarification(self, session_id: str, query: str):
        """开始澄清流程"""
        print(f"\n=== API请求 - 会话ID: {session_id} ===")
        print(f"用户问题: {query}")

        # 初始化会话
        self.active_sessions[session_id] = {
            'original_query': query,
            'conversation_history': [],
            'current_round': 0,
            'status': 'active'
        }

        # 第一轮分析
        result = self._process_round(session_id, query)
        return result

    def continue_clarification(self, session_id: str, user_answer: str):
        """继续澄清流程"""
        if session_id not in self.active_sessions:
            return {
                'status': 'error',
                'message': '会话不存在或已过期'
            }

        session = self.active_sessions[session_id]
        if session['status'] != 'active':
            return {
                'status': 'error',
                'message': '会话已结束'
            }

        print(f"\n=== API请求 - 会话ID: {session_id} 继续 ===")
        print(f"用户回答: {user_answer}")

        # 记录用户回答到最近的问题
        if session['conversation_history']:
            session['conversation_history'][-1]['user_answer'] = user_answer

        # 处理下一轮
        current_query = self._build_current_query(session)
        result = self._process_round(session_id, current_query)

        return result

    def _process_round(self, session_id: str, current_query: str):
        """处理单轮对话"""
        session = self.active_sessions[session_id]
        session['current_round'] += 1

        print(f"\n--- 第 {session['current_round']} 轮分析 ---")

        # 问题分类
        classification_result = self.clarifier.classifier.invoke(current_query)
        print(f"问题分类: {classification_result['classification']}")
        print(f"原因: {classification_result['reason']}")

        # 如果问题已经清晰或达到最大轮数，结束追问
        if (classification_result['classification'] == 'SIMPLE' or
                session['current_round'] >= self.clarifier.max_rounds):

            if classification_result['classification'] == 'SIMPLE':
                print("✅ 问题已经足够清晰，无需继续追问。")
            else:
                print(f"⚠️ 已达到最大追问轮数({self.clarifier.max_rounds})，结束追问。")

            # 生成最终结果
            final_query = self._generate_final_result(session_id)
            session['status'] = 'completed'

            return {
                'status': 'completed',
                'final_query': final_query,
                'session_id': session_id
            }

        # 确定策略并生成追问
        current_strategy = self.clarifier._determine_strategy(
            session['conversation_history'],
            classification_result
        )
        print(f"📋 当前策略: {self.clarifier._get_strategy_description(current_strategy)}")

        # 生成追问
        print("正在生成追问...")
        clarifying_question_result = self.clarifier.question_generator.invoke(
            current_query,
            f"{classification_result['reason']} | 当前需要: {self.clarifier._get_strategy_description(current_strategy)}"
        )
        question = clarifying_question_result['question']

        print(f"追问: {question}")

        # 记录对话历史
        session['conversation_history'].append({
            'round': session['current_round'],
            'strategy': current_strategy,
            'question': question,
            'user_answer': None  # 等待用户回答
        })

        return {
            'status': 'waiting_answer',
            'question': question,
            'round': session['current_round'],
            'session_id': session_id
        }

    def _build_current_query(self, session):
        """构建当前查询"""
        return self.clarifier._update_query_with_strategy(
            session['original_query'],
            session['conversation_history']
        )

    def _generate_final_result(self, session_id: str):
        """生成最终结果"""
        session = self.active_sessions[session_id]

        print("\n=== 生成最终结果 ===")

        # 生成最终查询
        final_query = self.clarifier._build_comprehensive_final_query(
            session['original_query'],
            session['conversation_history']
        )

        # 在控制台输出完整总结
        summary = self.clarifier._generate_final_summary(
            session['original_query'],
            session['conversation_history']
        )

        print("\n" + "=" * 60)
        print("🎯 最终总结:")
        print(summary)
        print("=" * 60)

        return final_query


# 创建服务实例
clarifier_service = APIClarifierService()


@app.route('/clarify/start', methods=['POST'])
def start_clarification():
    """开始澄清流程的API端点"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        query = data.get('query')

        if not session_id or not query:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数: session_id 和 query'
            }), 400

        result = clarifier_service.start_clarification(session_id, query)
        return jsonify(result)

    except Exception as e:
        print(f"API错误: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/clarify/continue', methods=['POST'])
def continue_clarification():
    """继续澄清流程的API端点"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_answer = data.get('answer')

        if not session_id or not user_answer:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数: session_id 和 answer'
            }), 400

        result = clarifier_service.continue_clarification(session_id, user_answer)
        return jsonify(result)

    except Exception as e:
        print(f"API错误: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    try:
        Config.validate()
        print("✅ 环境准备完成，DeepSeek API Key 已加载。")
        print("🚀 启动澄清服务API...")
        print("📍 API端点:")
        print("   - POST /clarify/start - 开始澄清流程")
        print("   - POST /clarify/continue - 继续澄清流程")
        print("   - GET /health - 健康检查")
        print("-" * 50)

        app.run(host='0.0.0.0', port=18890, debug=True)
    except ValueError as e:
        print(f"❌ 配置错误: {e}")