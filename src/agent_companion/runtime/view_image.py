"""只验证工作区中的图片；出站投影才会附上图片内容。"""

from typing import Any

from ..images import VIEW_IMAGE_OK, VIEW_IMAGE_TOOL_NAME, validate_workspace_image, view_image_path
from ..protocol import ToolSpec
from .tools import Tool, ToolContext


async def _view_image(args: dict[str, Any], context: ToolContext) -> str:
    path = view_image_path(args)
    if path is None:
        return '{"error":"invalid_arguments"}'
    if context.workspace is None or not await validate_workspace_image(context.workspace, path):
        return '{"error":"image_unavailable"}'
    return VIEW_IMAGE_OK


VIEW_IMAGE_TOOL = Tool(ToolSpec(
    VIEW_IMAGE_TOOL_NAME, "查看工作区相对路径指向的图片，包括用户上传的图片。",
    {"type": "object", "properties": {"path": {"type": "string", "description": "工作区相对文件路径"}},
     "required": ["path"], "additionalProperties": False}), _view_image)
