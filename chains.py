from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from models import Classification, QuestionGenerator, FinalQueryGenerator
from config import Config

class ClassifierChain:
    def __init__(self):
        self.model = ChatOpenAI(
            temperature=Config.TEMPERATURE,
            model_name=Config.MODEL_NAME,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_BASE_URL
        )
        self.parser = JsonOutputParser(pydantic_object=Classification)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.model | self.parser

    def _create_prompt(self):
        return ChatPromptTemplate.from_template(
            """
            你是一个专业的问题分析师。你的任务是判断用户的问题是否清晰明确，能够直接回答。

            分类标准：
            - SIMPLE: 问题表达清晰，有明确的询问对象和内容，虽然可能需要专业知识回答，但问题本身不模糊
            - COMPLEX: 问题涉及多个选择或比较，需要了解用户的具体偏好、预算、背景等才能给出个性化建议
            - VAGUE: 问题中包含模糊词汇或缺少关键信息，无法准确理解用户想问什么

            SIMPLE的例子：
            - "糖尿病可以吃炸鸡吗" - 问题明确，直接询问医学建议
            - "北京到上海的高铁票多少钱" - 问题具体，有明确查询对象
            - "Python怎么读取文件" - 技术问题明确

            COMPLEX的例子：
            - "去香港看病好还是在深圳看病好" - 需要了解用户关注的方面（费用/质量/便利性）
            - "买什么手机比较好" - 需要了解预算、用途、偏好

            VAGUE的例子：
            - "去香港哪家医院看好" - "好"的标准不明确
            - "这附近有什么餐厅" - "这附近"位置不明确
            - "怎么办" - 完全不知道要解决什么问题

            注意：不要因为问题需要专业知识回答就判断为COMPLEX，只要问题本身表达清晰就是SIMPLE。

            严格按照指示的JSON格式输出。

            {format_instructions}

            用户问题: "{query}"

            请分析这个问题并给出分类。
            """,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, query: str):
        return self.chain.invoke({"query": query})

class QuestionGeneratorChain:
    def __init__(self):
        self.model = ChatOpenAI(
            temperature=Config.TEMPERATURE,
            model_name=Config.MODEL_NAME,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_BASE_URL
        )
        self.parser = JsonOutputParser(pydantic_object=QuestionGenerator)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.model | self.parser

    def _create_prompt(self):
        return ChatPromptTemplate.from_template(
            """
            你是一个专业的医疗问询助手。一个用户提出了一个问题，但该问题被判断为模糊或复杂。
            你的任务是根据用户的原始问题和判断原因，生成1个具体的、有帮助的追问，以帮助用户明确他的需求。

            追问应该：
            1. 一次只问一个问题
            2. 自然、口语化
            3. 针对性强，直击问题核心
            4. 提供选项或引导方向

            严格按照指示的JSON格式输出。

            {format_instructions}

            用户的原始问题: "{query}"
            判断为模糊/复杂的原因: "{reason}"

            请生成一个追问。
            """,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, query: str, reason: str):
        return self.chain.invoke({"query": query, "reason": reason})

class FinalQueryGeneratorChain:
    def __init__(self):
        self.model = ChatOpenAI(
            temperature=Config.TEMPERATURE,
            model_name=Config.MODEL_NAME,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_BASE_URL
        )
        self.parser = JsonOutputParser(pydantic_object=FinalQueryGenerator)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.model | self.parser

    def _create_prompt(self):
        return ChatPromptTemplate.from_template(
            """
            你是一个专业的问题重构专家。用户原本提出了一个问题，通过多轮追问，我们收集了更多信息。
            现在你需要将这些信息整合，生成一个完整、清晰、自然的用户视角问题。

            要求：
            1. 从用户的角度重新表述问题
            2. 整合所有收集到的信息
            3. 表述自然、流畅，像真人提问
            4. 保持问题的核心意图不变
            5. 包含重要的背景信息和具体要求

            示例：
            原始问题: "在大陆看病有什么不好"
            追问收集到的信息: 用户关心卫生问题，想了解消毒措施、病房清洁度、医护人员卫生习惯
            最终问题: "想了解大陆医院在卫生方面的情况，包括医院的消毒措施、病房清洁度以及医护人员的卫生习惯如何？"

            另一个示例：
            原始问题: "去香港哪家医院看好"
            追问收集到的信息: 用户是糖尿病患者，关心专科医生水平和治疗费用
            最终问题: "作为糖尿病患者，想了解香港哪家医院的内分泌科专家比较好，治疗费用大概是什么水平？"

            严格按照指示的JSON格式输出。

            {format_instructions}

            对话历史信息:
            {conversation_summary}

            请生成最终的完整问题。
            """,
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, conversation_summary: str):
        return self.chain.invoke({"conversation_summary": conversation_summary})