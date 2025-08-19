from pydantic.v1 import BaseModel, Field

class Classification(BaseModel):
    classification: str = Field(description="问题的分类，必须是 'SIMPLE', 'COMPLEX', 或 'VAGUE' 中的一个。")
    reason: str = Field(description="做出该分类的简要原因。")

class QuestionGenerator(BaseModel):
    question: str = Field(description="生成的单个追问")

class FinalQueryGenerator(BaseModel):
    final_question: str = Field(description="整合了所有信息后生成的完整、自然的用户问题")