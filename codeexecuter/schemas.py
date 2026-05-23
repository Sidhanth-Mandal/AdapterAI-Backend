from typing import List , Any , Optional
from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Python type as a string, e.g. 'str', 'int', 'list', 'dict'")
    description: str = Field(default="", description="What this parameter represents")
    required: bool = Field(default=True, description="Whether parameter is required")
    example: Optional[Any] = Field(default=None,description="Example value for the parameter")


class OutputFieldSchema(BaseModel):

    name: str = Field(..., description="Output field name")
    type: str = Field(..., description="Output field type as a string, e.g. 'str', 'int', 'list', 'dict'")
    description: str = Field(..., description="Explanation of returned field")


class FunctionSchema(BaseModel):

    name: str = Field(..., description="Function name")
    description: str = Field(..., description="What the function does")
    parameters: List[ParameterSchema] = Field(default_factory=list, description="List of input parameters")
    outputs: List[OutputFieldSchema] = Field(default_factory=list, description="List of output fields")
    return_type: str = Field(default="dict", description="Return type as a plain string, e.g. 'dict', 'list', 'str'")


class ToolSchema(BaseModel):

    tool_name: str = Field(..., description="Unique tool name")
    tool_description: str = Field(..., description="High level explanation of what the tool does")
    category: str = Field(default="", description="Tool category, e.g. sports, weather, finance")
    dependencies: List[str] = Field(default_factory=list, description="Pip package names to install")
    functions: List[FunctionSchema] = Field(default_factory=list, description="Available tool functions")
    code: str = Field(..., description="Complete executable Python source code containing all the functions")
