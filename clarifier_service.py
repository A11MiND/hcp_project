from chains import ClassifierChain, QuestionGeneratorChain, FinalQueryGeneratorChain


class ClarifierService:
    def __init__(self, max_rounds=5):
        self.classifier = ClassifierChain()
        self.question_generator = QuestionGeneratorChain()
        self.final_query_generator = FinalQueryGeneratorChain()
        self.max_rounds = max_rounds

        # 通用的思考路径
        self.clarification_strategy = [
            "understand_intent",  # 第1步：确定用户真实需求/意图
            "gather_context",  # 第2步：收集用户背景信息
            "specify_details"  # 第3步：补充具体细节（如果需要）
        ]

    def run_clarifier_flow(self, user_query: str):
        """运行完整的澄清器流程"""
        print(f"--- 开始处理新问题: '{user_query}' ---")

        current_query = user_query
        round_count = 0
        conversation_history = []
        current_strategy = "initial"  # 追踪当前策略阶段

        while round_count < self.max_rounds:
            round_count += 1
            print(f"\n=== 第 {round_count} 轮分析 ===")

            # 问题分类
            classification_result = self.classifier.invoke(current_query)
            print(f"问题分类: {classification_result['classification']}")
            print(f"原因: {classification_result['reason']}")

            # 如果问题已经清晰，结束追问
            if classification_result['classification'] == 'SIMPLE':
                print("✅ 问题已经足够清晰，无需继续追问。")
                break

            # 如果达到最大轮数，强制结束
            if round_count >= self.max_rounds:
                print(f"⚠️ 已达到最大追问轮数({self.max_rounds})，结束追问。")
                break

            # 确定当前应该使用的策略
            current_strategy = self._determine_strategy(conversation_history, classification_result)
            print(f"📋 当前策略: {self._get_strategy_description(current_strategy)}")

            # 生成针对性追问
            print("正在生成追问...")
            clarifying_question_result = self.question_generator.invoke(
                current_query,
                f"{classification_result['reason']} | 当前需要: {self._get_strategy_description(current_strategy)}"
            )
            question = clarifying_question_result['question']

            print(f"追问: {question}")

            # 获取用户回答
            print(f"\n请回答上述问题 (轮次 {round_count}/{self.max_rounds}):")
            user_answer = input("> ").strip()

            if not user_answer:
                print("用户未提供回答，结束追问。")
                break

            # 记录对话历史
            conversation_history.append({
                'round': round_count,
                'strategy': current_strategy,
                'question': question,
                'user_answer': user_answer
            })

            # 智能更新当前查询
            current_query = self._update_query_with_strategy(user_query, conversation_history)
            print(f"已更新查询内容: {current_query}")

        # 生成最终总结
        final_summary = self._generate_final_summary(user_query, conversation_history)

        print("\n" + "=" * 60)
        print("🎯 最终总结:")
        print(final_summary)
        print("=" * 60)

        return final_summary

    def _determine_strategy(self, conversation_history: list, classification_result: dict):
        """确定当前应该使用的澄清策略"""
        if not conversation_history:
            return "understand_intent"

        # 检查是否已经理解了用户意图
        has_intent = any(conv['strategy'] == 'understand_intent' for conv in conversation_history)
        has_context = any(conv['strategy'] == 'gather_context' for conv in conversation_history)

        if not has_intent:
            return "understand_intent"
        elif not has_context and self._needs_user_context(conversation_history):
            return "gather_context"
        else:
            return "specify_details"

    def _needs_user_context(self, conversation_history: list):
        """判断是否需要收集用户背景信息"""
        # 检查用户意图是否涉及个人化建议或需要背景信息
        intent_answers = [conv['user_answer'] for conv in conversation_history
                          if conv['strategy'] == 'understand_intent']

        if not intent_answers:
            return True

        latest_intent = intent_answers[-1].lower()

        # 如果涉及个人建议、比较选择、治疗方案等，需要用户背景
        context_keywords = ['治疗', '看医生', '选择', '适合', '建议', '费用', '医院', '药物']
        return any(keyword in latest_intent for keyword in context_keywords)

    def _get_strategy_description(self, strategy: str):
        """获取策略描述"""
        descriptions = {
            "understand_intent": "明确用户的真实需求和意图",
            "gather_context": "收集用户背景信息和具体情况",
            "specify_details": "补充必要的具体细节"
        }
        return descriptions.get(strategy, "澄清问题")

    def _update_query_with_strategy(self, original_query: str, conversation_history: list):
        """根据策略和对话历史更新查询"""
        if not conversation_history:
            return original_query

        # 提取不同策略阶段的信息
        intent_info = self._extract_info_by_strategy(conversation_history, "understand_intent")
        context_info = self._extract_info_by_strategy(conversation_history, "gather_context")
        detail_info = self._extract_info_by_strategy(conversation_history, "specify_details")

        # 构建逐步完善的查询
        updated_parts = [f"用户原始问题: {original_query}"]

        if intent_info:
            updated_parts.append(f"用户真实需求: {intent_info}")

        if context_info:
            updated_parts.append(f"用户背景: {context_info}")

        if detail_info:
            updated_parts.append(f"具体要求: {detail_info}")

        return " | ".join(updated_parts)

    def _extract_info_by_strategy(self, conversation_history: list, strategy: str):
        """提取特定策略阶段的信息"""
        strategy_conversations = [conv for conv in conversation_history if conv['strategy'] == strategy]
        if not strategy_conversations:
            return ""

        # 合并同一策略下的所有回答
        answers = [conv['user_answer'] for conv in strategy_conversations]
        return "; ".join(answers)

    def _generate_final_summary(self, original_query: str, conversation_history: list):
        """生成最终总结"""
        if not conversation_history:
            return f"用户问题: {original_query}"

        summary = f"📋 对话总结:\n"
        summary += f"原始问题: {original_query}\n\n"

        # 按策略分组显示追问过程
        summary += f"追问过程:\n"
        for conv in conversation_history:
            strategy_desc = self._get_strategy_description(conv['strategy'])
            summary += f"第{conv['round']}轮 ({strategy_desc}): {conv['question']}\n"
            summary += f"用户回答: {conv['user_answer']}\n\n"

        # 使用大模型生成最终的自然问题
        final_query = self._build_comprehensive_final_query(original_query, conversation_history)
        summary += f"🎯 最终生成的用户问题:\n{final_query}"

        return summary

    def _build_comprehensive_final_query(self, original_query: str, conversation_history: list):
        """使用大模型构建自然的最终查询"""
        if not conversation_history:
            return original_query

        # 准备对话历史信息
        conversation_summary = self._prepare_conversation_summary(original_query, conversation_history)

        try:
            # 调用大模型生成自然的最终问题
            result = self.final_query_generator.invoke(conversation_summary)
            return result['final_question']
        except Exception as e:
            print(f"生成最终问题时出错: {e}")
            # 如果大模型调用失败，返回一个基本的汇总
            return self._build_fallback_final_query(original_query, conversation_history)

    def _prepare_conversation_summary(self, original_query: str, conversation_history: list):
        """准备对话历史的摘要信息"""
        summary = f"原始问题: {original_query}\n\n追问过程:\n"

        for conv in conversation_history:
            strategy_desc = self._get_strategy_description(conv['strategy'])
            summary += f"- {strategy_desc}: {conv['question']}\n"
            summary += f"  用户回答: {conv['user_answer']}\n"

        return summary

    def _build_fallback_final_query(self, original_query: str, conversation_history: list):
        """当大模型调用失败时的备用方案"""
        # 提取各个维度的信息
        intent = self._extract_info_by_strategy(conversation_history, "understand_intent")
        context = self._extract_info_by_strategy(conversation_history, "gather_context")
        details = self._extract_info_by_strategy(conversation_history, "specify_details")

        # 构建基本的最终查询
        parts = [original_query]

        if context:
            parts.append(f"（用户背景：{context}）")

        if intent:
            parts.append(f"具体想了解：{intent}")

        if details:
            parts.append(f"关注细节：{details}")

        return "，".join(parts)